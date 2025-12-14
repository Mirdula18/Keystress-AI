"""
Typing Data Collection Module for Keystress-AI

This module captures typing behavior metadata WITHOUT storing actual
typed content. It focuses on timing patterns for burnout detection.

Privacy Note: This module is designed to be privacy-friendly.
It does NOT store or log the actual characters typed by users.
Only timing metadata is collected.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TypingSession:
    """
    Stores metadata about a typing session.
    
    Attributes:
        timestamps: List of Unix timestamps for each key press
        is_backspace: List indicating if each key was a backspace
        start_time: Session start timestamp
        end_time: Session end timestamp
    """
    timestamps: List[float] = field(default_factory=list)
    is_backspace: List[bool] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def record_keypress(self, is_backspace: bool = False):
        """
        Record a single key press event.
        
        Parameters:
            is_backspace (bool): Whether the key pressed was backspace
        """
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            
        self.timestamps.append(current_time)
        self.is_backspace.append(is_backspace)
        self.end_time = current_time
    
    def get_inter_key_delays(self) -> List[float]:
        """
        Calculate time delays between consecutive key presses.
        
        Returns:
            List[float]: List of inter-key delays in seconds
        """
        if len(self.timestamps) < 2:
            return []
        
        delays = []
        for i in range(1, len(self.timestamps)):
            delay = self.timestamps[i] - self.timestamps[i-1]
            delays.append(delay)
        
        return delays
    
    def get_total_keys(self) -> int:
        """Get total number of keys pressed."""
        return len(self.timestamps)
    
    def get_backspace_count(self) -> int:
        """Get total number of backspace keys pressed."""
        return sum(self.is_backspace)
    
    def get_duration(self) -> float:
        """
        Get total typing duration in seconds.
        
        Returns:
            float: Duration of typing session
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def reset(self):
        """Reset the typing session for a new recording."""
        self.timestamps = []
        self.is_backspace = []
        self.start_time = None
        self.end_time = None


class TypingDataCollector:
    """
    Collects typing metadata from user input.
    
    This class provides methods to capture typing behavior
    without storing the actual content typed.
    """
    
    def __init__(self):
        """Initialize the typing data collector."""
        self.session = TypingSession()
    
    def start_session(self):
        """Start a new typing session."""
        self.session.reset()
    
    def record_key(self, is_backspace: bool = False):
        """
        Record a key press event.
        
        Parameters:
            is_backspace (bool): Whether the key was backspace
        """
        self.session.record_keypress(is_backspace)
    
    def end_session(self) -> TypingSession:
        """
        End the current typing session.
        
        Returns:
            TypingSession: The completed session data
        """
        self.session.end_time = time.time()
        return self.session
    
    def get_session_data(self) -> dict:
        """
        Get typing session metadata as a dictionary.
        
        Returns:
            dict: Dictionary containing typing metadata
        """
        return {
            'total_keys': self.session.get_total_keys(),
            'backspace_count': self.session.get_backspace_count(),
            'duration': self.session.get_duration(),
            'inter_key_delays': self.session.get_inter_key_delays(),
            'start_time': self.session.start_time,
            'end_time': self.session.end_time
        }


def process_keystroke_data(keystroke_events: List[dict]) -> dict:
    """
    Process raw keystroke events from web interface.
    
    This function takes timestamped keystroke events and extracts
    typing metadata without storing the actual characters.
    
    Parameters:
        keystroke_events: List of dicts with 'timestamp' and 'is_backspace' keys
        
    Returns:
        dict: Processed typing metadata
    """
    if not keystroke_events:
        return {
            'total_keys': 0,
            'backspace_count': 0,
            'duration': 0,
            'inter_key_delays': [],
            'start_time': None,
            'end_time': None
        }
    
    timestamps = [event['timestamp'] for event in keystroke_events]
    is_backspace = [event.get('is_backspace', False) for event in keystroke_events]
    
    # Calculate inter-key delays
    inter_key_delays = []
    for i in range(1, len(timestamps)):
        delay = timestamps[i] - timestamps[i-1]
        inter_key_delays.append(delay)
    
    return {
        'total_keys': len(timestamps),
        'backspace_count': sum(is_backspace),
        'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
        'inter_key_delays': inter_key_delays,
        'start_time': timestamps[0],
        'end_time': timestamps[-1]
    }


if __name__ == "__main__":
    # Demo of the typing data collection
    print("Typing Data Collection Module Demo")
    print("=" * 50)
    
    collector = TypingDataCollector()
    collector.start_session()
    
    # Simulate some key presses
    print("Simulating key presses...")
    for i in range(10):
        time.sleep(0.2)  # Simulate typing delay
        is_backspace = (i == 5)  # Simulate one backspace
        collector.record_key(is_backspace)
    
    session_data = collector.get_session_data()
    
    print(f"\nSession Data:")
    print(f"  Total keys: {session_data['total_keys']}")
    print(f"  Backspaces: {session_data['backspace_count']}")
    print(f"  Duration: {session_data['duration']:.2f}s")
    print(f"  Avg delay: {sum(session_data['inter_key_delays'])/len(session_data['inter_key_delays']):.3f}s")
