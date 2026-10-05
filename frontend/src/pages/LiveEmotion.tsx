import React, { useEffect, useRef, useState } from 'react';
import api from '../services/api';
import { getSocket } from '../services/socket';
import { EmotionPrediction, EmotionLog, SessionContext, EmotionExplanation } from '../types';
import { EmotionOverlay } from '../components/EmotionOverlay';
import { LiveChart } from '../components/LiveChart';
import { ValenceArousalChart } from '../components/ValenceArousalChart';
import { EmotionExplainerModal } from '../components/EmotionExplainerModal';
import { soundManager } from '../utils/audioFeedback';
import { 
  Video, 
  Square, 
  Play, 
  GraduationCap, 
  Stethoscope, 
  Headphones, 
  AlertCircle, 
  RefreshCw,
  Sparkles,
  Zap,
  Activity,
  Compass,
  Upload
} from 'lucide-react';

export const LiveEmotion: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [context, setContext] = useState<SessionContext>('education');
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<EmotionPrediction | null>(null);
  const [logs, setLogs] = useState<EmotionLog[]>([]);
  const [webcamError, setWebcamError] = useState<string | null>(null);

  // AI Engine Selection State
  const [aiEngine, setAiEngine] = useState<'local' | 'gemini' | 'openai'>('local');
  const [apiKey, setApiKey] = useState<string>(() => localStorage.getItem('emosense_llm_api_key') || '');
  const [showKeyModal, setShowKeyModal] = useState<boolean>(false);

  // Explainer modal state
  const [explainerOpen, setExplainerOpen] = useState<boolean>(false);
  const [explainerData, setExplainerData] = useState<EmotionExplanation | null>(null);
  const [explainerLoading, setExplainerLoading] = useState<boolean>(false);

  const intervalRef = useRef<any>(null);
  const lastEmotionRef = useRef<string | null>(null);
  const aiEngineRef = useRef(aiEngine);
  const apiKeyRef = useRef(apiKey);

  // Keep refs in sync with state so the interval callback sees the latest values
  useEffect(() => { aiEngineRef.current = aiEngine; }, [aiEngine]);
  useEffect(() => { apiKeyRef.current = apiKey; }, [apiKey]);

  const handleSaveApiKey = (key: string) => {
    setApiKey(key);
    localStorage.setItem('emosense_llm_api_key', key);
    setShowKeyModal(false);
    soundManager.playSuccessChime();
  };

  // Initialize Socket.io listener
  useEffect(() => {
    const socket = getSocket();
    if (!socket.connected) socket.connect();

    socket.on('emotionUpdate', (data: any) => {
      setPrediction({
        emotion: data.emotion,
        confidence: data.confidence,
        all_probs: data.all_probs,
        bbox: data.bbox,
        timestamp: data.timestamp
      });

      // Sound feedback on state transition
      if (lastEmotionRef.current && lastEmotionRef.current !== data.emotion) {
        soundManager.playBlip(540, 40);
      }
      lastEmotionRef.current = data.emotion;

      const newLog: EmotionLog = {
        sessionId: data.sessionId,
        timestamp: data.timestamp || new Date().toISOString(),
        emotion: data.emotion,
        confidence: data.confidence,
        all_probs: data.all_probs
      };

      setLogs((prev) => [...prev, newLog]);
    });

    socket.on('emotionError', (err: any) => {
      console.warn('Socket emotion error:', err);
    });

    return () => {
      socket.off('emotionUpdate');
      socket.off('emotionError');
    };
  }, []);

  // Request Webcam Stream with multi-tier fallback and clean track release
  const startWebcam = async () => {
    // 1. Ensure any previous stream tracks are completely stopped
    stopWebcam();
    setWebcamError(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const unsupportedMsg = 'Camera API (getUserMedia) is not supported in this browser or context.';
      setWebcamError(unsupportedMsg);
      throw new Error(unsupportedMsg);
    }

    let stream: MediaStream | null = null;

    // Tier 1: Try with ideal resolution and front facing mode
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
    } catch (err1: any) {
      console.warn('Initial camera constraint failed, retrying with basic video constraint...', err1);
      // Tier 2: Fallback to basic video constraint if specific resolution or facingMode is locked
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (err2: any) {
        let msg = 'Unable to access webcam.';
        if (err2.name === 'NotReadableError' || err2.message?.includes('Device in use')) {
          msg = 'Webcam is currently in use by another application or browser tab. Please close other apps/tabs using the camera and click "Start Live Camera" again.';
        } else if (err2.name === 'NotAllowedError' || err2.name === 'PermissionDeniedError') {
          msg = 'Camera permission was denied. Please click the lock/settings icon in your browser address bar and allow camera access.';
        } else if (err2.name === 'NotFoundError' || err2.name === 'DevicesNotFoundError') {
          msg = 'No webcam device found on your system. You can test emotion detection using "Test with Image Upload".';
        } else {
          msg = `Unable to access webcam: ${err2.message || err2.name || 'Unknown error'}.`;
        }
        setWebcamError(msg);
        throw new Error(msg);
      }
    }

    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch (playErr) {
        console.warn('Video element play() was delayed:', playErr);
      }
    }
  };

  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      try {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => {
          track.stop();
          track.enabled = false;
        });
      } catch (e) {
        console.warn('Error stopping stream tracks:', e);
      }
      videoRef.current.srcObject = null;
    }
  };

  // Start Session & Frame Broadcast
  const handleStartSession = async () => {
    try {
      setWebcamError(null);
      await startWebcam();
      soundManager.playSuccessChime();
      const res = await api.post('/sessions', { context });
      const session = res.data;
      setActiveSessionId(session._id);
      setIsStreaming(true);
      setLogs([]);

      const socket = getSocket();
      socket.emit('startSession', { sessionId: session._id });

      // Adaptive frame rate: 800ms for cloud Groq Vision, 2500ms for heavy multimodal LLMs
      const frameInterval = aiEngine === 'local' ? 800 : 2500;
      intervalRef.current = setInterval(() => {
        captureAndSendFrame(session._id);
      }, frameInterval);
    } catch (err: any) {
      console.error('Failed to start live session:', err);
      stopWebcam();
      setIsStreaming(false);
    }
  };

  // Image Upload Fallback for static testing
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64Img = event.target?.result as string;
      
      try {
        setWebcamError(null);
        // Clear previous prediction immediately
        setPrediction(null);
        soundManager.playSuccessChime();
        
        const res = await api.post('/emotions/predict-frame', {
          image: base64Img,
          frame: base64Img,
          engine: aiEngineRef.current !== 'local' ? aiEngineRef.current : undefined,
          provider: aiEngineRef.current !== 'local' ? aiEngineRef.current : undefined,
          apiKey: aiEngineRef.current !== 'local' && apiKeyRef.current ? apiKeyRef.current : undefined
        });
        
        const data = res.data;
        if (data && data.emotion) {
          setPrediction({
            emotion: data.emotion,
            confidence: data.confidence,
            all_probs: data.all_probs,
            bbox: data.bbox || null
          });
        }
      } catch (err: any) {
        setWebcamError(`Image prediction failed: ${err.message}`);
      }
    };
    reader.readAsDataURL(file);
    
    // Reset input so the same file can be uploaded again if needed
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleStopSession = async () => {
    soundManager.playBlip(440, 70);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    stopWebcam();
    setIsStreaming(false);

    if (activeSessionId) {
      try {
        await api.post(`/sessions/${activeSessionId}/end`);
        const socket = getSocket();
        socket.emit('stopSession', { sessionId: activeSessionId });
      } catch (err) {
        console.error('Failed to end session:', err);
      }
    }
  };

  const captureAndSendFrame = async (sessionId: string) => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || video.videoWidth === 0) return;

    if (!canvasRef.current) {
      canvasRef.current = document.createElement('canvas');
    }
    const canvas = canvasRef.current;
    // Scale to 480x360 with 0.82 quality for high-fidelity facial landmark and smile detection
    canvas.width = 480;
    canvas.height = 360;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, 480, 360);
    const base64Frame = canvas.toDataURL('image/jpeg', 0.82);

    const socket = getSocket();
    if (socket && socket.connected) {
      socket.emit('frame', { 
        sessionId, 
        frame: base64Frame,
        engine: aiEngineRef.current !== 'local' ? aiEngineRef.current : undefined,
        apiKey: aiEngineRef.current !== 'local' && apiKeyRef.current ? apiKeyRef.current : undefined
      });
    } else {
      // Direct HTTP Fallback if socket is reconnecting
      try {
        const res = await api.post('/emotions/predict-frame', {
          image: base64Frame,
          sessionId,
          provider: aiEngineRef.current !== 'local' ? aiEngineRef.current : undefined,
          apiKey: aiEngineRef.current !== 'local' && apiKeyRef.current ? apiKeyRef.current : undefined
        });
        const data = res.data;
        setPrediction({
          emotion: data.emotion,
          confidence: data.confidence,
          all_probs: data.all_probs,
          bbox: data.bbox || [30, 20, 180, 200],
          timestamp: new Date().toISOString()
        });
        setLogs(prev => [...prev, {
          sessionId,
          timestamp: new Date().toISOString(),
          emotion: data.emotion,
          confidence: data.confidence,
          all_probs: data.all_probs
        }]);
      } catch (httpErr) {
        console.warn('HTTP frame fallback error:', httpErr);
      }
    }
  };

  const handleExplainLive = async () => {
    soundManager.playBlip(700, 60);
    setExplainerOpen(true);
    setExplainerLoading(true);

    try {
      const currentEmo = prediction?.emotion || 'neutral';
      const conf = prediction?.confidence || 0.85;
      const res = await api.post('/emotions/explain', {
        emotion: currentEmo,
        confidence: conf,
        all_probs: prediction?.all_probs,
        context,
        totalLogs: logs.length
      });
      setExplainerData(res.data);
    } catch (err) {
      console.error('Explanation request failed:', err);
    } finally {
      setExplainerLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      stopWebcam();
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white flex items-center gap-2.5">
            <div className="p-2 rounded-2xl bg-rose-500/20 text-rose-400 border border-rose-500/30 shadow-md">
              <Video className="w-6 h-6" />
            </div>
            Live Emotion Recognition
          </h1>
          <p className="text-xs text-slate-400">
            Real-time facial expression telemetry via Socket.io frame streaming & OpenCV CNN inference
          </p>
        </div>

        {/* Controls & Context Picker */}
        <div className="flex flex-wrap items-center gap-3">
          {/* AI Engine Switcher */}
          <div className="flex bg-slate-900 border border-slate-800 rounded-2xl p-1">
            <button
              onClick={() => setAiEngine('local')}
              disabled={isStreaming}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                aiEngine === 'local' ? 'bg-indigo-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              Local SE-ResNet
            </button>
            <button
              onClick={() => {
                setAiEngine('gemini');
                if (!apiKey) setShowKeyModal(true);
              }}
              disabled={isStreaming}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                aiEngine === 'gemini' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🧠 Gemini Vision
            </button>
            <button
              onClick={() => {
                setAiEngine('openai');
                if (!apiKey) setShowKeyModal(true);
              }}
              disabled={isStreaming}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                aiEngine === 'openai' ? 'bg-emerald-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              🤖 GPT-4o
            </button>
            <button
              onClick={() => setShowKeyModal(true)}
              title="Configure API Key"
              className="p-1.5 rounded-xl text-slate-400 hover:text-amber-400 transition-colors ml-1"
            >
              🔑
            </button>
          </div>

          {/* Context Picker */}
          <div className="flex bg-slate-900 border border-slate-800 rounded-2xl p-1">
            <button
              onClick={() => setContext('education')}
              disabled={isStreaming}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                context === 'education' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-inner' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <GraduationCap className="w-3.5 h-3.5" />
              Education
            </button>
            <button
              onClick={() => setContext('healthcare')}
              disabled={isStreaming}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                context === 'healthcare' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-inner' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Stethoscope className="w-3.5 h-3.5" />
              Healthcare
            </button>
            <button
              onClick={() => setContext('customer')}
              disabled={isStreaming}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition ${
                context === 'customer' ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-inner' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Headphones className="w-3.5 h-3.5" />
              Customer
            </button>
          </div>

        </div>
      </div>

      {/* Action Buttons Row */}
      <div className="flex flex-col sm:flex-row items-center gap-4 mt-6">
        {!isStreaming ? (
          <button
            onClick={handleStartSession}
            className="flex items-center justify-center gap-2 px-6 py-3 rounded-2xl bg-gradient-to-r from-emerald-600 to-teal-600 font-bold text-sm text-white shadow-xl shadow-emerald-600/30 hover:opacity-95 hover:scale-102 transition w-full sm:w-auto"
          >
            <Play className="w-4 h-4 fill-white" />
            Start Live Camera
          </button>
        ) : (
          <button
            onClick={handleStopSession}
            className="flex items-center justify-center gap-2 px-6 py-3 rounded-2xl bg-gradient-to-r from-rose-600 to-red-600 font-bold text-sm text-white shadow-xl shadow-rose-600/30 hover:opacity-95 hover:scale-102 transition w-full sm:w-auto"
          >
            <Square className="w-4 h-4 fill-white" />
            Stop Session
          </button>
        )}

        <div className="text-slate-500 font-medium text-xs uppercase hidden sm:block">OR</div>

        {/* Static Image Upload Fallback */}
        <input 
          id="live-fallback-upload"
          name="live-fallback-upload"
          type="file" 
          accept="image/*" 
          ref={fileInputRef} 
          onChange={handleImageUpload} 
          className="hidden" 
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="flex items-center justify-center gap-2 px-6 py-3 rounded-2xl bg-slate-800 border border-slate-700 hover:border-indigo-500/50 hover:bg-slate-800/80 font-bold text-sm text-indigo-300 shadow-xl transition w-full sm:w-auto"
        >
          <Upload className="w-4 h-4" />
          Test with Image Upload
        </button>
      </div>

      {webcamError && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-2xl flex items-center gap-3 text-rose-300 text-xs">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{webcamError}</span>
        </div>
      )}

      {/* Main Video & Probability Gauges Section */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Webcam Viewport with Overlay HUD */}
        <div className="lg:col-span-2 glass-panel p-4 rounded-3xl border border-slate-800 relative overflow-hidden flex flex-col items-center justify-center min-h-[440px]">
          <div className="relative w-full max-w-2xl aspect-video rounded-2xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex items-center justify-center">
            <video
              ref={videoRef}
              playsInline
              muted
              className="w-full h-full object-cover transform -scale-x-100"
            />
            {isStreaming && (
              <EmotionOverlay
                prediction={prediction}
                videoWidth={videoRef.current?.videoWidth || 640}
                videoHeight={videoRef.current?.videoHeight || 480}
              />
            )}
            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-sm text-center p-6 space-y-3">
                <div className="p-4 rounded-3xl bg-slate-900 border border-slate-800 text-slate-500 shadow-inner">
                  <Video className="w-10 h-10" />
                </div>
                <div className="text-sm font-bold text-slate-200">Live Camera Feed Inactive</div>
                <p className="text-xs text-slate-500 max-w-sm">
                  Click "Start Live Session" above to activate your browser webcam and begin streaming frame telemetry.
                </p>
              </div>
            )}
          </div>

          {/* Quick Explainer & Status Bar underneath Video */}
          <div className="w-full max-w-2xl mt-4 flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <span className={`w-2.5 h-2.5 rounded-full ${isStreaming ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
              <span>{isStreaming ? `Tracking session: ${activeSessionId?.slice(-6)}` : 'Camera ready'}</span>
            </div>

            <button
              onClick={handleExplainLive}
              disabled={!prediction && !isStreaming}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-bold text-xs shadow-md shadow-indigo-600/30 hover:opacity-90 transition disabled:opacity-40"
            >
              <Sparkles className="w-3.5 h-3.5 text-amber-300" />
              <span>Explain Emotion with AI</span>
            </button>
          </div>
        </div>

        {/* 7-Class Softmax Probability Breakdown Card */}
        <div className="glass-panel p-5 rounded-3xl border border-slate-800 flex flex-col justify-between space-y-4">
          <div>
            <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center justify-between">
              <span>7-Class Softmax Probabilities</span>
              {isStreaming && (
                <span className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-semibold">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse-glow" />
                  Live ~300ms
                </span>
              )}
            </h3>

            {prediction && prediction.all_probs ? (
              <div className="space-y-2.5">
                {Object.entries(prediction.all_probs).map(([emo, prob]) => {
                  const pct = Math.round(prob * 100);
                  const isDominant = prediction.emotion === emo;
                  return (
                    <div key={emo} className="space-y-1">
                      <div className="flex justify-between text-xs font-semibold capitalize text-slate-300">
                        <span className={isDominant ? 'text-indigo-400 font-bold' : ''}>{emo}</span>
                        <span className={isDominant ? 'text-white font-bold' : 'text-slate-400'}>{pct}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-300 ${
                            isDominant 
                              ? 'bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500' 
                              : 'bg-slate-700'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="h-64 flex flex-col items-center justify-center text-slate-500 text-xs italic text-center p-4">
                <RefreshCw className="w-6 h-6 mb-2 animate-spin text-slate-600" />
                Awaiting active facial telemetry stream...
              </div>
            )}
          </div>

          <div className="p-3.5 bg-slate-900/80 rounded-2xl border border-slate-800 text-[11px] text-slate-400 space-y-1">
            <div className="flex justify-between">
              <span>Context: <span className="text-indigo-300 font-bold capitalize">{context}</span></span>
              <span>Logged Frames: <span className="text-white font-bold">{logs.length}</span></span>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Charts Row: Live Timeline Chart + 2D Valence-Arousal Plane */}
      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <LiveChart logs={logs} height={260} />
        </div>

        <div className="glass-panel p-5 rounded-3xl border border-slate-800 flex flex-col justify-between">
          <div className="mb-2">
            <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Compass className="w-4 h-4 text-purple-400" />
              Live Valence-Arousal Position
            </h4>
            <p className="text-[10px] text-slate-500">
              Russell's Circumplex quadrant coordinates updated in real time.
            </p>
          </div>

          <ValenceArousalChart 
            currentEmotion={prediction?.emotion || 'neutral'} 
            height={200} 
          />
        </div>
      </div>

      {/* Emotion Explainer Modal */}
      <EmotionExplainerModal
        isOpen={explainerOpen}
        onClose={() => setExplainerOpen(false)}
        explanation={explainerData}
        loading={explainerLoading}
      />

      {/* API Key Configuration Modal */}
      {showKeyModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fade-in">
          <div className="glass-panel p-6 sm:p-8 rounded-3xl border border-slate-700 bg-slate-900 max-w-md w-full space-y-5 shadow-2xl relative">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <span>🔑</span>
                <span>Configure Vision LLM API Key</span>
              </h3>
              <button 
                onClick={() => setShowKeyModal(false)}
                className="text-slate-400 hover:text-white text-sm font-bold"
              >
                ✕
              </button>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Enter your <strong>Google Gemini API Key</strong> or <strong>OpenAI API Key</strong> for live webcam emotion detection powered by multimodal Vision AI.
            </p>
            <div className="space-y-2">
              <label className="text-xs font-bold text-slate-300 uppercase tracking-wider">API Key</label>
              <input
                id="live-api-key"
                name="live-api-key"
                type="password"
                defaultValue={apiKey}
                placeholder="AIzaSy... or sk-proj-..."

                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2.5 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setShowKeyModal(false)}
                className="px-4 py-2 rounded-xl text-xs font-bold text-slate-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const input = document.getElementById('live-api-key-input') as HTMLInputElement;
                  if (input) handleSaveApiKey(input.value.trim());
                }}
                className="btn-primary px-5 py-2 rounded-xl text-xs font-bold"
              >
                Save Key
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
