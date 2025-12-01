import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sounddevice as sd
import threading
import time
from typing import Optional, List


class AudioPlaybackPlot:
    """
    Interactive audio playback synchronized with matplotlib plots.
    
    Usage:
        fig, ax = plot_vap(waveform, p_now, p_fut, vad)
        player = AudioPlaybackPlot(fig, ax, waveform, sample_rate=16000)
        plt.show()
    """
    
    def __init__(
        self,
        fig: plt.Figure,
        axes: List[plt.Axes],
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        frame_hz: int = 50,
        indicator_color: str = 'red',
        indicator_linewidth: float = 2.5,
        indicator_alpha: float = 0.8,
    ):
        """
        Args:
            fig: Matplotlib figure
            axes: List of axes to add playback indicator to
            waveform: Audio waveform tensor, shape (2, n_samples) or (n_samples,)
            sample_rate: Audio sample rate in Hz
            frame_hz: Frame rate for predictions (used for time scaling)
            indicator_color: Color of the playback position indicator
            indicator_linewidth: Width of the indicator line
            indicator_alpha: Transparency of the indicator line
        """
        self.fig = fig
        
        # Handle different axes formats
        if isinstance(axes, np.ndarray):
            # Flatten if it's a numpy array of axes
            self.axes = axes.flatten().tolist()
        elif isinstance(axes, list):
            # If it's already a list, keep it
            self.axes = axes
        elif hasattr(axes, '__iter__'):
            # If it's any other iterable, convert to list
            self.axes = list(axes)
        else:
            # Single axis, wrap in list
            self.axes = [axes]
        
        # Filter out any non-Axes objects
        self.axes = [ax for ax in self.axes if isinstance(ax, plt.Axes)]
        
        if not self.axes:
            raise ValueError("No valid matplotlib Axes found in axes parameter")
        
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        
        # Handle different waveform shapes
        if waveform.ndim == 2 and waveform.shape[0] == 2:
            # Stereo: convert to mono for playback
            self.audio = waveform.mean(0).cpu().numpy() * 2.0
        elif waveform.ndim == 1:
            self.audio = waveform.cpu().numpy() * 2.0
        else:
            raise ValueError(f"Unexpected waveform shape: {waveform.shape}")
        
        self.duration = len(self.audio) / sample_rate
        
        # Playback state
        self.is_playing = False
        self.current_time = 0.0
        self.start_time = None
        self.pause_time = 0.0
        
        # Create vertical line indicators on all axes
        self.indicators = []
        for ax in self.axes:
            line = ax.axvline(
                x=0,
                color=indicator_color,
                linewidth=indicator_linewidth,
                alpha=indicator_alpha,
                zorder=100,
                label='Playback Position'
            )
            self.indicators.append(line)
        
        # Add control buttons
        self._create_controls()
        
        # Animation
        self.anim = None
        self._setup_animation()
        
    def _create_controls(self):
        """Create play/pause and reset buttons."""
        # Adjust figure to make room for buttons
        self.fig.subplots_adjust(bottom=0.15)
        
        # Play/Pause button
        ax_play = self.fig.add_axes([0.3, 0.02, 0.15, 0.05])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        # Reset button
        ax_reset = self.fig.add_axes([0.55, 0.02, 0.15, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset)
        
    def _setup_animation(self):
        """Setup the animation for updating the indicator."""
        def update(frame):
            if self.is_playing:
                elapsed = time.time() - self.start_time
                self.current_time = self.pause_time + elapsed
                
                if self.current_time >= self.duration:
                    self._stop()
                    self.current_time = self.duration
            
            # Convert time to frame index
            current_frame = self.current_time * self.frame_hz
            
            # Update all indicators
            # First axis uses time (waveform), others use frame indices
            for i, indicator in enumerate(self.indicators):
                if i == 0:
                    # First plot (waveform) uses time in seconds
                    indicator.set_xdata([self.current_time])
                else:
                    # Other plots use frame indices
                    indicator.set_xdata([current_frame])
            
            return self.indicators
        
        self.anim = animation.FuncAnimation(
            self.fig,
            update,
            interval=20,  # Update every 20ms for smooth animation
            blit=True,
            cache_frame_data=False
        )
    
    def _toggle_play(self, event):
        """Toggle between play and pause."""
        if not self.is_playing:
            self._play()
        else:
            self._pause()
    
    def _play(self):
        """Start audio playback and animation."""
        if self.current_time >= self.duration:
            self.current_time = 0.0
            self.pause_time = 0.0
        
        self.is_playing = True
        self.btn_play.label.set_text('Pause')
        
        # Start audio playback from current position
        start_sample = int(self.current_time * self.sample_rate)
        audio_segment = self.audio[start_sample:]
        
        # Use sounddevice's callback for better sync
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback,
            finished_callback=self._audio_finished
        )
        
        self.audio_position = 0
        self.audio_segment = audio_segment
        self.start_time = time.time()
        
        self.stream.start()
        self.fig.canvas.draw_idle()
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback for streaming playback."""
        if status:
            print(f"Audio status: {status}")
        
        end_pos = self.audio_position + frames
        chunk = self.audio_segment[self.audio_position:end_pos]
        
        if len(chunk) < frames:
            outdata[:len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            raise sd.CallbackStop()
        else:
            outdata[:, 0] = chunk
        
        self.audio_position = end_pos
    
    def _audio_finished(self):
        """Called when audio finishes playing."""
        self._stop()
    
    def _pause(self):
        """Pause audio playback and animation."""
        self.is_playing = False
        self.pause_time = self.current_time
        self.btn_play.label.set_text('Play')
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.fig.canvas.draw_idle()
    
    def _stop(self):
        """Stop audio playback."""
        self.is_playing = False
        self.btn_play.label.set_text('Play')
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.fig.canvas.draw_idle()
    
    def _reset(self, event):
        """Reset playback to the beginning."""
        self._pause()
        self.current_time = 0.0
        self.pause_time = 0.0
        for i, indicator in enumerate(self.indicators):
            if i == 0:
                indicator.set_xdata([0])
            else:
                indicator.set_xdata([0])  # Frame 0
        self.fig.canvas.draw_idle()


# Convenience function to add to existing plots
def add_audio_playback(
    fig: plt.Figure,
    axes: List[plt.Axes],
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    frame_hz: int = 50,
) -> AudioPlaybackPlot:
    """
    Add interactive audio playback to existing VAP plots.
    
    Example:
        fig, ax = plot_vap(waveform, p_now, p_fut, vad)
        player = add_audio_playback(fig, ax, waveform)
        plt.show()
    """
    return AudioPlaybackPlot(fig, axes, waveform, sample_rate, frame_hz)