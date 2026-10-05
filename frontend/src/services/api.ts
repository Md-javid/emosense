import axios, { AxiosError } from 'axios';

const API_BASE_URL = 
  import.meta.env.VITE_API_URL || 
  (typeof window !== 'undefined' && window.location.hostname !== 'localhost'
    ? 'https://emosense-esz2.onrender.com/api'
    : 'http://localhost:5000/api');

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 35000 // 35s timeout to gracefully allow Render free tier wakeups
});

// Inject Authorization header dynamically
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('emosense_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Auto-retry interceptor on Render Free Tier Cold Starts (500, 502, 503, 504, or Network Timeout)
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const config = error.config as any;
    if (!config) return Promise.reject(error);

    config.__retryCount = config.__retryCount || 0;
    const maxRetries = 2;

    const isColdStartOrTimeout = 
      error.code === 'ECONNABORTED' || 
      !error.response || 
      (error.response.status >= 500 && error.response.status <= 504);

    if (isColdStartOrTimeout && config.__retryCount < maxRetries) {
      config.__retryCount += 1;
      const delayMs = config.__retryCount * 2500;
      console.warn(`[API Auto-Retry] Cold-start/delay detected (${error.message}). Retrying request ${config.__retryCount}/${maxRetries} in ${delayMs}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
      return api(config);
    }

    return Promise.reject(error);
  }
);

export default api;
