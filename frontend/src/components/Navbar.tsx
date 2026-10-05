import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { soundManager } from '../utils/audioFeedback';
import { 
  Sun, 
  Moon, 
  ShieldAlert, 
  LogOut, 
  User as UserIcon, 
  BrainCircuit, 
  Volume2, 
  VolumeX, 
  Menu, 
  X,
  Sparkles
} from 'lucide-react';
import { Link } from 'react-router-dom';

interface NavbarProps {
  onOpenPrivacy: () => void;
  onToggleMobileMenu?: () => void;
  isMobileMenuOpen?: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ 
  onOpenPrivacy, 
  onToggleMobileMenu,
  isMobileMenuOpen = false 
}) => {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [isMuted, setIsMuted] = useState(soundManager.getMuted());
  const [engineStatus, setEngineStatus] = useState<'ready' | 'waking' | 'checking'>('checking');

  // Automated background keep-alive ping to prevent Render free-tier cold starts
  React.useEffect(() => {
    let isMounted = true;
    const pingServices = async () => {
      try {
        const backendBase = (import.meta.env.VITE_API_URL || (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1' ? 'https://emosense-esz2.onrender.com/api' : 'http://localhost:5000/api')).replace(/\/api\/?$/, '');
        const res = await fetch(`${backendBase}/health`, { method: 'GET' });
        if (res.ok && isMounted) {
          setEngineStatus('ready');
        } else if (isMounted) {
          setEngineStatus('waking');
        }
      } catch (err) {
        if (isMounted) setEngineStatus('waking');
      }
    };

    pingServices();
    // Ping every 3.5 minutes while tab is open to prevent container sleep
    const interval = setInterval(pingServices, 210000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleToggleSound = () => {
    const muted = soundManager.toggleMute();
    setIsMuted(muted);
    if (!muted) {
      soundManager.playBlip(680, 70);
    }
  };

  const getRoleBadgeColor = (role?: string) => {
    switch (role) {
      case 'teacher': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
      case 'therapist': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
      case 'agent': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
      case 'admin': return 'bg-rose-500/20 text-rose-400 border-rose-500/30';
      default: return 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30';
    }
  };

  return (
    <header className="sticky top-0 z-40 w-full glass-panel border-b border-slate-800/90">
      <div className="flex h-16 items-center justify-between px-4 sm:px-6">
        {/* Left Side: Mobile Menu Button & Brand */}
        <div className="flex items-center gap-3">
          {onToggleMobileMenu && (
            <button
              onClick={onToggleMobileMenu}
              className="p-2 rounded-xl md:hidden border border-slate-800 bg-slate-900/60 text-slate-300 hover:text-white"
              title="Toggle Menu"
            >
              {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          )}

          <Link to="/dashboard" className="flex items-center gap-3 group">
            <div className="relative">
              <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-indigo-600 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/30 group-hover:scale-105 transition-transform">
                <BrainCircuit className="w-6 h-6 text-white" />
              </div>
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-slate-950 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="text-xl font-extrabold bg-gradient-to-r from-white via-indigo-200 to-indigo-400 bg-clip-text text-transparent">
                  EmoSense
                </span>
                <span className="text-[10px] font-black uppercase px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  AI v2.0
                </span>
              </div>
              <span className="block text-[10px] text-slate-400 tracking-wider font-medium uppercase">
                Affective Emotion Intelligence
              </span>
            </div>
          </Link>
        </div>

        {/* Right Side Actions */}
        <div className="flex items-center gap-2.5 sm:gap-3">
          {/* Audio Feedback Toggle */}
          <button
            onClick={handleToggleSound}
            className={`p-2 rounded-xl border transition ${
              isMuted
                ? 'border-slate-800 text-slate-500 hover:text-slate-300'
                : 'border-indigo-500/40 bg-indigo-500/10 text-indigo-300 shadow-sm'
            }`}
            title={isMuted ? "Unmute Audio Cues" : "Mute Audio Cues"}
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>

          {/* Cloud AI Status Badge */}
          <div className={`hidden md:flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1.5 rounded-xl border ${
            engineStatus === 'ready' 
              ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30' 
              : 'bg-amber-500/10 text-amber-300 border-amber-500/30 animate-pulse'
          }`}>
            <span className={`w-2 h-2 rounded-full ${engineStatus === 'ready' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            <span>{engineStatus === 'ready' ? 'Cloud AI Online' : 'Waking Cloud AI...'}</span>
          </div>

          {/* Privacy Modal Trigger */}
          <button
            onClick={onOpenPrivacy}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 px-3 py-2 rounded-xl border border-slate-800 hover:bg-slate-800/60 transition"
            title="Privacy & Data Safeguards"
          >
            <ShieldAlert className="w-4 h-4 text-emerald-400" />
            <span className="hidden sm:inline font-medium">Privacy</span>
          </button>

          {/* Theme Switcher */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl border border-slate-800 hover:bg-slate-800/60 text-slate-300 transition"
            title="Toggle Light/Dark Mode"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-indigo-600" />}
          </button>

          {/* User Profile / Logout */}
          {user && (
            <div className="flex items-center gap-2 pl-2 sm:pl-3 border-l border-slate-800">
              <Link to="/profile" className="flex items-center gap-2 group p-1 rounded-xl hover:bg-slate-800/40 transition">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-slate-800 to-indigo-950 border border-slate-700 flex items-center justify-center text-indigo-300 font-bold text-xs group-hover:border-indigo-500 transition">
                  {user.name ? user.name.charAt(0).toUpperCase() : <UserIcon className="w-4 h-4" />}
                </div>
                <div className="hidden lg:block text-left">
                  <div className="text-xs font-bold text-slate-200 max-w-[110px] truncate">{user.name || user.email}</div>
                  <span className={`inline-block text-[9px] uppercase font-extrabold px-1.5 rounded border ${getRoleBadgeColor(user.role)}`}>
                    {user.role}
                  </span>
                </div>
              </Link>

              <button
                onClick={logout}
                className="p-2 rounded-xl hover:bg-rose-500/10 hover:text-rose-400 text-slate-400 transition"
                title="Sign Out"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
