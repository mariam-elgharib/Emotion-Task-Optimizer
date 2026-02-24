# gui_app.py - Modern Integrated GUI with Camera
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import pickle
import os
import sys
from datetime import datetime, timedelta
import time
import cv2
from PIL import Image, ImageTk
import numpy as np

# Import backend modules
sys.path.append('.')
from task import Task
from default_tasks import get_default_mood_tasks
from algorithms import (create_preference_task, csp_preferences, mini_a_star, 
                        greedy, stochastic, hill_climbing, csp_task)

# Set appearance and color theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TaskOptimizerGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🎭 Emotion Task Optimizer")
        self.geometry("1550x850")  # Wider for side panel
        
        # Initialize camera and detector
        self.cap = None
        self.detector = None  # Will be initialized when needed
        self.camera_running = False
        self.current_emotion = "neutral"
        self.last_emotion = "neutral"  # Track previous emotion
        self.last_algo_update = 0
        self.recommendation_results = []
        self.last_emotion_seen = None

        # Initialize data
        self.todo_tasks = []
        self.mood_activities = []
        self.csp_preferences = []
        self.algorithm_choice = ctk.IntVar(value=3)  # Default to CSP
        self.use_deadline_warnings = ctk.BooleanVar(value=True)
        
        # Session State
        self.is_session_active = False
        self.active_task = None
        self.interrupted_task = None
        self.session_start_time = 0
        self.task_start_time = 0
        self.task_elapsed_before_pause = 0  # Track elapsed time before pause
        self.completed_tasks = []
        self.break_suggestion_active = False
        self.suggested_break_task = None
        
        # Timer variables
        self.todo_timer_running = False
        self.todo_time_remaining = 0  # in seconds
        self.todo_total_duration = 0  # in seconds
        self.break_timer_running = False
        self.break_time_remaining = 0  # in seconds
        
        # Resources from entry
        self.available_resources = ["computer", "internet", "water"]
        
        # Create detector only when needed
        self.create_detector()
        
        # Grid configuration - 3 columns: sidebar, main, recommendations
        self.grid_columnconfigure(1, weight=1)  # Main content
        self.grid_columnconfigure(2, weight=0)  # Recommendations panel
        self.grid_rowconfigure(0, weight=1)

        # Initialize UI components first
        self.sidebar_frame = None
        self.logo_label = None
        self.sidebar_status = None
        self.tabview = None
        self.recommendations_frame = None
        
        # Setup the UI
        self.setup_ui()

    def setup_ui(self):
        """Setup the complete UI"""
        # Sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=180, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Logo and title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Emotion Task Optimizer", 
                                      font=ctk.CTkFont(size=18, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        # Navigation buttons
        nav_buttons = [
            ("📊 Dashboard", "Dashboard"),
            ("📋 Must-Do Tasks", "Must-Do Tasks"),
            ("😊 Mood Activities", "Mood Activities"),
            ("⚙️ Preferences", "Preferences"),
            ("📹 Live Camera", "Live Camera"),
            ("🚀 Start Session", "Start Session"),
            ("⏱️ Active Session", "Active Session")
        ]
        
        for i, (text, tab_name) in enumerate(nav_buttons, 1):
            btn = ctk.CTkButton(self.sidebar_frame, text=text, 
                               command=lambda tn=tab_name: self.tabview.set(tn),
                               anchor="w", height=40)
            btn.grid(row=i, column=0, padx=15, pady=5, sticky="ew")
        
        # Status label
        self.sidebar_status = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", 
                                          text_color="gray", font=ctk.CTkFont(size=12))
        self.sidebar_status.grid(row=7, column=0, padx=20, pady=20)
        
        # Main Tabview (middle)
        self.tabview = ctk.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=1, padx=(20, 0), pady=(20, 20), sticky="nsew")
        
        # Add tabs
        tabs = ["Dashboard", "Must-Do Tasks", "Mood Activities", "Preferences", 
                "Live Camera", "Start Session", "Active Session"]
        for tab in tabs:
            self.tabview.add(tab)
            self.tabview.tab(tab).grid_columnconfigure(0, weight=1)
        
        # Setup all tabs
        self.setup_dashboard()
        self.setup_tasks_tab()
        self.setup_mood_tab()
        self.setup_preferences_tab()
        self.setup_camera_tab()
        self.setup_start_tab()
        self.setup_active_session_tab()
        
        # Recommendations Panel (right side) - Setup AFTER all tabs
        self.setup_recommendations_panel()

        self.load_saved_data()
        self.update_stats()
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start emotion monitoring thread
        self.start_emotion_monitoring()

    def create_detector(self):
        """Create emotion detector on demand"""
        try:
            from emotion_detector import EmotionDetector
            self.detector = EmotionDetector()
        except Exception as e:
            print(f"Warning: Could not initialize emotion detector: {e}")
            self.detector = None

    def start_emotion_monitoring(self):
        """Start thread to monitor emotion changes"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.emotion_monitor_loop, daemon=True)
        self.monitor_thread.start()

    def emotion_monitor_loop(self):
        """Monitor for emotion changes and update recommendations"""
        while getattr(self, 'monitoring', True):
            if self.detector and self.camera_running and self.cap:
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        emotion = self.detector.detect_emotion_from_frame(frame)
                        
                        # Only update if emotion actually changed
                        if emotion != self.current_emotion:
                            self.current_emotion = emotion
                            self.last_emotion_seen = emotion
                            
                            # Update UI labels
                            self.after(0, self.update_emotion_labels, emotion)
                            
                            # Update recommendations if session is active
                            if self.is_session_active:
                                self.after(0, self.update_recommendations_panel)
                except Exception as e:
                    print(f"Error in emotion detection: {e}")
                    pass
            
            time.sleep(1)  # Check every second

    def update_emotion_labels(self, emotion):
        """Update all emotion labels in the UI"""
        if hasattr(self, 'emotion_label') and self.emotion_label:
            self.emotion_label.configure(text=f"Emotion: {emotion.upper()}")
        if hasattr(self, 'start_emotion_label') and self.start_emotion_label:
            self.start_emotion_label.configure(text=f"Current Emotion: {emotion.upper()}", text_color="#1abc9c")
        if hasattr(self, 'session_emotion_label') and self.session_emotion_label:
            self.session_emotion_label.configure(text=f"Feeling: {emotion.upper()}", text_color="#1abc9c")
        if hasattr(self, 'recommendation_emotion_label') and self.recommendation_emotion_label:
            self.recommendation_emotion_label.configure(text=f"Current Mood: {emotion.upper()}", text_color="#1abc9c")
        
        self.update_stats()

    def setup_recommendations_panel(self):
        """Setup the right-side recommendations panel"""
        self.recommendations_frame = ctk.CTkFrame(self, width=350, corner_radius=10)
        self.recommendations_frame.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        self.recommendations_frame.grid_propagate(False)
        
        # Title
        title = ctk.CTkLabel(self.recommendations_frame, text="✨ Recommendations", 
                            font=ctk.CTkFont(size=20, weight="bold"))
        title.pack(pady=(20, 10), padx=20)
        
        # Current emotion display
        self.recommendation_emotion_label = ctk.CTkLabel(self.recommendations_frame, 
                                                         text="Current Mood: neutral",
                                                         font=ctk.CTkFont(size=16),
                                                         text_color="#1abc9c")
        self.recommendation_emotion_label.pack(pady=(0, 20), padx=20)
        
        # Algorithm info
        self.recommendation_algo_label = ctk.CTkLabel(self.recommendations_frame,
                                                     text="Algorithm: CSP",
                                                     font=ctk.CTkFont(size=12),
                                                     text_color="gray")
        self.recommendation_algo_label.pack(pady=(0, 10), padx=20)
        
        # Separator
        separator = ctk.CTkFrame(self.recommendations_frame, height=2, fg_color="gray")
        separator.pack(fill="x", padx=20, pady=10)
        
        # Recommendations scrollable area
        self.recommendations_scroll = ctk.CTkScrollableFrame(self.recommendations_frame, height=600)
        self.recommendations_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Initial message
        self.recommendation_container = self.recommendations_scroll
        self.show_no_recommendations_message()

    def show_no_recommendations_message(self):
        """Show message when no recommendations available"""
        if not hasattr(self, 'recommendation_container'):
            return
            
        for widget in self.recommendation_container.winfo_children():
            widget.destroy()
        
        label = ctk.CTkLabel(self.recommendation_container, 
                            text="No recommendations yet.\nStart a session and emotions will trigger suggestions.",
                            text_color="gray",
                            justify="center")
        label.pack(pady=50)

    def update_recommendations_panel(self):
        """Update the recommendations panel with current algorithm results"""
        if not hasattr(self, 'recommendation_container'):
            return
            
        if not self.is_session_active or not self.current_emotion:
            self.show_no_recommendations_message()
            return
        
        # Get current recommendations (mood tasks only, not must-do tasks)
        recommendations = self.get_current_algorithm_recs(include_must_do=False)
        
        # Update algorithm label
        algo_map = {
            1: "Mini A*",
            2: "Greedy", 
            3: "CSP",
            4: "Stochastic",
            5: "Hill Climbing"
        }
        if hasattr(self, 'recommendation_algo_label'):
            self.recommendation_algo_label.configure(
                text=f"Algorithm: {algo_map.get(self.algorithm_choice.get(), 'CSP')}"
            )
        
        # Clear current recommendations
        for widget in self.recommendation_container.winfo_children():
            widget.destroy()
        
        if not recommendations:
            ctk.CTkLabel(self.recommendation_container, 
                        text=f"No mood tasks for '{self.current_emotion}' emotion.",
                        text_color="gray").pack(pady=20)
            return
        
        # Show recommendations
        ctk.CTkLabel(self.recommendation_container,
                    text=f"Suggested for {self.current_emotion}:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(0, 10))
        
        for i, task in enumerate(recommendations[:5], 1):  # Show top 5
            self.create_recommendation_widget(task, i)

    def create_recommendation_widget(self, task, index):
        """Create a recommendation widget with action button"""
        if not hasattr(self, 'recommendation_container'):
            return
            
        frame = ctk.CTkFrame(self.recommendation_container, 
                            fg_color="#2a2a2a",
                            border_width=1,
                            border_color="#444")
        frame.pack(fill="x", padx=5, pady=5)
        
        # Algorithm badge and Sequence pos
        badge_frame = ctk.CTkFrame(frame, fg_color="transparent")
        badge_frame.pack(fill="x", padx=10, pady=(5, 0))
        
        algo_tag = getattr(task, 'recommended_by', "AI")
        badge_color = "#3498db" # Default blue
        if algo_tag == "CSP": badge_color = "#9b59b6" # Purple
        elif algo_tag == "A*": badge_color = "#f1c40f" # Yellow/Gold
        elif algo_tag == "Greedy": badge_color = "#2ecc71" # Green
        elif algo_tag == "Hill Climb": badge_color = "#e74c3c" # Red
        
        tag_label = ctk.CTkLabel(badge_frame, text=f" {algo_tag} ", 
                               font=ctk.CTkFont(size=10, weight="bold"),
                               fg_color=badge_color, text_color="white", corner_radius=4)
        tag_label.pack(side="left")
        
        if hasattr(task, 'sequence_pos'):
            pos_labels = {1: "1st Suggestion", 2: "2nd Suggestion", 3: "3rd Suggestion"}
            pos_label = ctk.CTkLabel(badge_frame, text=f"  {pos_labels.get(task.sequence_pos, f'Pos {task.sequence_pos}')}", 
                                   font=ctk.CTkFont(size=10, slant="italic"),
                                   text_color="gray")
            pos_label.pack(side="left")
        
        # Task info
        info_frame = ctk.CTkFrame(frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        # Task name and duration
        name_label = ctk.CTkLabel(info_frame, 
                                 text=f"{task.name}",
                                 font=ctk.CTkFont(size=14, weight="bold"),
                                 anchor="w",
                                 justify="left")
        name_label.pack(anchor="w")
        
        # Duration and emotion fit
        details = f"⏱️ {task.duration} min"
        if hasattr(task, 'emotion_fit') and task.emotion_fit:
            emotions = ", ".join(task.emotion_fit[:3])
            details += f" | 😊 {emotions}"
        
        details_label = ctk.CTkLabel(info_frame, 
                                    text=details,
                                    text_color="gray",
                                    anchor="w",
                                    justify="left")
        details_label.pack(anchor="w", pady=(2, 0))
        
        # Constraints info
        if hasattr(task, 'constraints'):
            constraints = task.constraints
            if "requires" in constraints:
                req = constraints["requires"]
                if isinstance(req, list):
                    req_text = ", ".join(req)
                else:
                    req_text = str(req)
                if req_text:
                    req_label = ctk.CTkLabel(info_frame,
                                           text=f"📦 Requires: {req_text}",
                                           text_color="lightblue",
                                           font=ctk.CTkFont(size=11),
                                           anchor="w")
                    req_label.pack(anchor="w", pady=(2, 0))
        
        # Action button
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Only show "Take Break" button if we're currently working on a todo task
        if self.active_task and hasattr(self.active_task, 'task_type') and self.active_task.task_type == "must_do":
            btn_text = "⏸️ Take Break"
            btn_command = lambda t=task: self.start_break_task(t)
            btn_color = "#e67e22"  # Orange for break
            hover_color = "#d35400"
        else:
            btn_text = "✨ Start Activity"
            btn_command = lambda t=task: self.start_break_task(t)
            btn_color = "#1abc9c"  # Green for regular start
            hover_color = "#16a085"
        
        action_btn = ctk.CTkButton(btn_frame,
                                  text=btn_text,
                                  command=btn_command,
                                  fg_color=btn_color,
                                  hover_color=hover_color,
                                  height=35)
        action_btn.pack(fill="x")

    def start_break_task(self, task):
        """Start a mood-changing task from recommendations"""
        # 1. STOP THE WORK TIMER FIRST
        self.stop_todo_timer()
        
        # Check if we're in a session
        if not self.is_session_active:
            messagebox.showinfo("Info", "Please start a session first!")
            return
        
        # Check if we're currently working on a todo task
        if self.active_task and hasattr(self.active_task, 'task_type') and self.active_task.task_type == "must_do":
            # Store the current state for resumption later
            # self.todo_time_remaining is already updated by the timer loop
            self.task_elapsed_before_pause = self.todo_total_duration - self.todo_time_remaining
            self.interrupted_task = self.active_task
            
            # Update UI to show paused state
            self.update_paused_state()
        
        # Start the mood task
        self.active_task = task
        self.task_start_time = time.time()
        
        # Start break timer
        self.start_break_timer(task.duration * 60)  # Convert minutes to seconds
        
        # Stop emotion detection during break
        if self.camera_running:
            self.camera_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
        
        # Update UI
        self.tabview.set("Active Session")
        if hasattr(self, 'active_task_name'):
            self.active_task_name.configure(text=f"Break: {self.active_task.name}")
        if hasattr(self, 'active_task_details'):
            self.active_task_details.configure(text=f"Goal: Relax | Duration: {self.active_task.duration}min")
        if hasattr(self, 'session_emotion_label'):
            self.session_emotion_label.configure(text="Feeling: ON BREAK", text_color="#e67e22")
        
        # Update timer display
        self.update_timer_display()
        
        # Show success message
        messagebox.showinfo("Break Started", 
                          f"Starting: {task.name}\n\n"
                          f"Duration: {task.duration} minutes\n"
                          f"After this break, you'll return to your previous task.")

    def start_todo_timer(self, duration_minutes):
        """Start countdown timer for todo task"""
        self.todo_timer_running = True
        self.todo_total_duration = duration_minutes * 60  # Convert to seconds
        self.todo_time_remaining = self.todo_total_duration
        
        # Start the timer loop
        self.update_todo_timer()

    def stop_todo_timer(self):
        """Stop the todo timer"""
        self.todo_timer_running = False

    def update_todo_timer(self):
        """Update the todo timer countdown"""
        if self.todo_timer_running and self.todo_time_remaining > 0:
            self.todo_time_remaining -= 1
            
            # Update display
            self.update_timer_display()
            
            # Check if timer finished
            if self.todo_time_remaining <= 0:
                self.todo_timer_complete()
            else:
                # Schedule next update in 1 second
                self.after(1000, self.update_todo_timer)
        elif self.todo_time_remaining <= 0:
            self.todo_timer_complete()

    def todo_timer_complete(self):
        """Handle todo timer completion"""
        self.todo_timer_running = False
        if hasattr(self, 'timer_label'):
            self.timer_label.configure(text="00:00:00", text_color="red")
        
        # Mark task as completed
        if self.active_task:
            self.completed_tasks.append(self.active_task)
            messagebox.showinfo("Task Complete", f"Time's up! Completed: {self.active_task.name}")
            
            # Move to next task
            self.complete_current_task()

    def start_break_timer(self, duration_seconds):
        """Start countdown timer for break task"""
        self.break_timer_running = True
        self.break_time_remaining = duration_seconds
        
        # Start the timer loop
        self.update_break_timer()

    def stop_break_timer(self):
        """Stop the break timer"""
        self.break_timer_running = False

    def update_break_timer(self):
        """Update the break timer countdown"""
        if self.break_timer_running and self.break_time_remaining > 0:
            self.break_time_remaining -= 1
            
            # Update display
            self.update_timer_display()
            
            # Check if timer finished
            if self.break_time_remaining <= 0:
                self.break_timer_complete()
            else:
                # Schedule next update in 1 second
                self.after(1000, self.update_break_timer)
        elif self.break_time_remaining <= 0:
            self.break_timer_complete()

    def break_timer_complete(self):
        """Handle break timer completion"""
        self.break_timer_running = False
        self.end_break_task()

    def update_timer_display(self):
        """Update the timer display based on active timer"""
        if hasattr(self, 'timer_label'):
            if self.break_timer_running:
                # Show break timer
                time_str = self.format_time(self.break_time_remaining)
                self.timer_label.configure(text=time_str, text_color="#e67e22")
            elif self.todo_timer_running:
                # Show todo timer
                time_str = self.format_time(self.todo_time_remaining)
                self.timer_label.configure(text=time_str, text_color="#1abc9c")
            else:
                # No active timer
                self.timer_label.configure(text="00:00:00", text_color="gray")

    def update_paused_state(self):
        """Update UI to show todo task is paused"""
        if self.interrupted_task and hasattr(self, 'active_task_name') and hasattr(self, 'active_task_details'):
            elapsed_str = self.format_time(self.task_elapsed_before_pause)
            total_str = self.format_time(self.todo_total_duration)
            self.active_task_name.configure(text=f"⏸️ PAUSED: {self.interrupted_task.name}")
            self.active_task_details.configure(text=f"Elapsed: {elapsed_str} | Total: {total_str}")

    def format_time(self, seconds):
        """Format seconds to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def end_break_task(self):
        """End the break and return to interrupted task"""
        # CRITICAL: Stop break timer flag so display can switch back to todo
        self.stop_break_timer()
        
        if self.interrupted_task:
            # Return to the interrupted todo task
            self.active_task = self.interrupted_task
            self.interrupted_task = None
            
            # Use current remaining time (already tracked)
            remaining_time = self.todo_time_remaining
            
            # RESUME todo timer loop
            self.todo_timer_running = True
            self.update_todo_timer()
            
            # Restart camera for emotion detection
            self.start_camera()
            
            # Update UI
            if hasattr(self, 'active_task_name'):
                self.active_task_name.configure(text=f"Current Task: {self.active_task.name}")
            if hasattr(self, 'active_task_details'):
                # Calculate elapsed for display
                elapsed = self.todo_total_duration - self.todo_time_remaining
                elapsed_str = self.format_time(elapsed)
                total_str = self.format_time(self.todo_total_duration)
                self.active_task_details.configure(text=f"Duration: {self.active_task.duration}m | Elapsed: {elapsed_str}/{total_str}")
            if hasattr(self, 'session_emotion_label'):
                self.session_emotion_label.configure(text="Feeling: WORKING", text_color="#1abc9c")
            
            # Update timer display immediately
            self.update_timer_display()
            
            messagebox.showinfo("Break Ended", 
                              f"Welcome back! Resuming: {self.active_task.name}\n"
                              f"Time remaining: {self.format_time(remaining_time)}")
        else:
            # No interrupted task, go back to task selection
            self.active_task = None
            self.tabview.set("Start Session")
            
            # Restart camera
            self.start_camera()
            
            # Reset timer display
            if hasattr(self, 'timer_label'):
                self.timer_label.configure(text="00:00:00", text_color="gray")
            
            messagebox.showinfo("Break Ended", "Break completed! Select a new task to continue.")

    def get_current_algorithm_recs(self, include_must_do=True):
        """Get recommendations using the selected algorithm"""
        algo_choice = self.algorithm_choice.get()
        emotion = self.current_emotion or "neutral"
        
        conditions = {
            "current_time": datetime.now(),
            "available_resources": self.available_resources,
            "current_energy": 5,
            "current_emotion": emotion
        }
        
        # Get active mood activities (default tasks)
        active_moods = []
        if hasattr(self, 'mood_vars'):
            active_moods = [task for var, task in self.mood_vars if var.get()]
            for t in active_moods:
                t.is_preference = False # Reset flag for default tasks
        
        # Set preference flag ONLY for manual CSP preferences
        for t in self.csp_preferences:
            t.is_preference = True
        
        unified_pool = active_moods + self.csp_preferences
        
        recommendations = []
        
        # CSP Algorithm
        if algo_choice == 3:
            # Mood tasks only (no must-do tasks in recommendations panel)
            if not include_must_do and unified_pool:
                try:
                    pref_recs, _ = csp_preferences(unified_pool, conditions)
                    recommendations.extend(pref_recs[:5])
                except:
                    pass
            
            # Must-do tasks for active session
            elif include_must_do and self.todo_tasks:
                undone = [t for t in self.todo_tasks if t not in self.completed_tasks]
                try:
                    rec, warning, urgency = csp_task(undone, conditions["current_time"])
                    if rec:
                        recommendations.append(rec)
                except:
                    pass
        
        # Other algorithms
        else:
            # Mood tasks
            if not include_must_do and unified_pool:
                if algo_choice == 1:  # A*
                    try:
                        # Pass conditions to mini_a_star
                        recs = mini_a_star([], emotion, conditions, 3, unified_pool)
                        if recs:
                            recommendations.extend(recs[:3])
                            # Mark sequence for A*
                            for i, task in enumerate(recommendations):
                                task.sequence_pos = i + 1
                    except Exception as e:
                        print(f"A* Error: {e}")
                elif algo_choice == 2:  # Greedy
                    try:
                        best = greedy(unified_pool, emotion, conditions)
                        if best:
                            recommendations.append(best)
                    except Exception as e:
                        print(f"Greedy Error: {e}")
                elif algo_choice == 4:  # Stochastic
                    try:
                        best = stochastic(unified_pool, emotion, conditions)
                        if best:
                            recommendations.append(best)
                    except Exception as e:
                        print(f"Stochastic Error: {e}")
                elif algo_choice == 5:  # Hill Climbing
                    try:
                        best = hill_climbing(unified_pool, conditions)
                        if best:
                            recommendations.append(best)
                    except Exception as e:
                        print(f"Hill Climbing Error: {e}")
            
            # Must-do tasks
            elif include_must_do and self.todo_tasks:
                undone = [t for t in self.todo_tasks if t not in self.completed_tasks]
                for task in undone:
                    try:
                        task.compute_score(emotion, conditions["current_time"])
                    except:
                        pass
                undone.sort(key=lambda t: getattr(t, 'score', 0), reverse=True)
                if undone:
                    recommendations.append(undone[0])
        
        # Tag recommendations with algorithm type for UI display
        algo_names = {1: "A*", 2: "Greedy", 3: "CSP", 4: "Stochastic", 5: "Hill Climb"}
        curr_algo_name = algo_names.get(algo_choice, "AI")
        for task in recommendations:
            task.recommended_by = curr_algo_name
            
        return recommendations

    def toggle_camera(self):
        """Start/stop the camera"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start the camera feed"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        self.camera_running = True
        if hasattr(self, 'camera_btn'):
            self.camera_btn.configure(text="⏹️ Stop Camera")
        self.update_camera()

    def update_camera(self):
        """Update camera frame - simplified, emotion detection handled in monitor thread"""
        if self.camera_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                
                # Add current emotion text
                cv2.putText(frame, f"Emotion: {self.current_emotion}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert to ImageTk
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update labels
                if hasattr(self, 'camera_label'):
                    self.camera_label.configure(image=imgtk, text="")
                    self.camera_label.image = imgtk
                
                if hasattr(self, 'session_camera_label'):
                    self.session_camera_label.configure(image=imgtk, text="")
                    self.session_camera_label.image = imgtk
                
                # Schedule next update
                self.after(10, self.update_camera)
            else:
                self.stop_camera()
        else:
            self.stop_camera()

    def stop_camera(self):
        """Stop the camera feed"""
        self.camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if hasattr(self, 'camera_btn'):
            self.camera_btn.configure(text="▶️ Start Camera")
        
        if hasattr(self, 'camera_label'):
            self.camera_label.configure(image="", text="Camera stopped")
        
        if hasattr(self, 'emotion_label'):
            self.emotion_label.configure(text="Emotion: --")

    def start_gui_session(self):
        """Launch the session directly in the GUI"""
        if not self.todo_tasks:
            messagebox.showwarning("Warning", "Please add some must-do tasks first!")
            return
        
        self.is_session_active = True
        self.session_start_time = time.time()
        self.completed_tasks = []
        
        # Get resources from entry if it exists
        if hasattr(self, 'resources_entry'):
            res_text = self.resources_entry.get()
            if res_text:
                self.available_resources = [r.strip() for r in res_text.split(',')]
        
        # Pick first task
        undone = [t for t in self.todo_tasks if t not in self.completed_tasks]
        if not undone:
            messagebox.showinfo("Info", "All tasks already completed!")
            self.stop_gui_session()
            return
        
        # Sort by deadline if exists, otherwise by priority
        for task in undone:
            if hasattr(task, 'deadline') and task.deadline:
                try:
                    datetime.strptime(task.deadline, "%Y-%m-%d")
                except:
                    task.deadline = None
        
        undone.sort(key=lambda t: (
            t.deadline if hasattr(t, 'deadline') and t.deadline else "9999-12-31",
            -getattr(t, 'base_priority', 0)
        ))
        
        self.active_task = undone[0]
        self.task_start_time = time.time()
        
        # START the todo timer
        self.start_todo_timer(self.active_task.duration)
        
        # Update UI
        self.tabview.set("Active Session")
        if hasattr(self, 'session_title'):
            self.session_title.configure(text="🚀 Session Active")
        if hasattr(self, 'active_task_name'):
            self.active_task_name.configure(text=f"Current Task: {self.active_task.name}")
        if hasattr(self, 'active_task_details'):
            self.active_task_details.configure(text=f"Duration: {self.active_task.duration}m | Priority: {getattr(self.active_task, 'base_priority', 0)}")
        if hasattr(self, 'sidebar_status'):
            self.sidebar_status.configure(text="Status: SESSION ACTIVE", text_color="#1abc9c")
        
        # Update timer display
        self.update_timer_display()
        
        # Start camera for emotion detection
        self.start_camera()
        
        # Start session loop
        self.run_session_loop()
        
        # Update recommendations panel
        self.update_recommendations_panel()

    def complete_current_task(self):
        """Mark task as done and pick next"""
        if not self.active_task:
            return
        
        # Check if it's a mood task (break)
        if hasattr(self.active_task, 'task_type') and self.active_task.task_type != "must_do" and self.interrupted_task:
            # This is a break, return to interrupted task
            self.end_break_task()
            return
        
        # It's a must-do task, mark as completed
        self.completed_tasks.append(self.active_task)
        
        # Pick next must-do
        undone = [t for t in self.todo_tasks if t not in self.completed_tasks]
        if not undone:
            messagebox.showinfo("Session Over", "All tasks completed! Amazing work.")
            self.stop_gui_session()
            return
        
        # Sort by deadline if exists, otherwise by priority
        for task in undone:
            if hasattr(task, 'deadline') and task.deadline:
                try:
                    datetime.strptime(task.deadline, "%Y-%m-%d")
                except:
                    task.deadline = None
        
        undone.sort(key=lambda t: (
            t.deadline if hasattr(t, 'deadline') and t.deadline else "9999-12-31",
            -getattr(t, 'base_priority', 0)
        ))
        
        self.active_task = undone[0]
        self.task_start_time = time.time()
        
        # START new todo timer for the new task
        self.start_todo_timer(self.active_task.duration)
        
        # Update UI
        if hasattr(self, 'active_task_name'):
            self.active_task_name.configure(text=f"Current Task: {self.active_task.name}")
        if hasattr(self, 'active_task_details'):
            self.active_task_details.configure(text=f"Duration: {self.active_task.duration}m | Priority: {getattr(self.active_task, 'base_priority', 0)}")
        
        # Update timer display
        self.update_timer_display()
        
        messagebox.showinfo("Task Completed", f"Great job! Moving to next task: {self.active_task.name}")

    def setup_dashboard(self):
        """Setup dashboard with statistics"""
        tab = self.tabview.tab("Dashboard")
        tab.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Title
        title = ctk.CTkLabel(tab, text="📊 Dashboard Overview", 
                            font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, columnspan=3, padx=20, pady=(20, 30))
        
        # Stats frame
        stats_frame = ctk.CTkFrame(tab)
        stats_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Statistics widgets
        self.stats_widgets = {}
        stats_data = [
            ("Must-Do Tasks", "0", "📝"),
            ("Mood Activities", "0", "😊"),
            ("Preferences", "0", "⚙️"),
            ("Current Emotion", "neutral", "🎭")
        ]
        
        for i, (title_text, value, icon) in enumerate(stats_data):
            frame = ctk.CTkFrame(stats_frame, height=100)
            frame.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            frame.grid_propagate(False)
            
            ctk.CTkLabel(frame, text=icon, font=ctk.CTkFont(size=24)).pack(pady=(15, 5))
            ctk.CTkLabel(frame, text=title_text, font=ctk.CTkFont(size=14, weight="bold")).pack()
            value_label = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=28, weight="bold"))
            value_label.pack(pady=(5, 15))
            
            self.stats_widgets[title_text] = value_label
        
        # Algorithm info
        algo_frame = ctk.CTkFrame(tab)
        algo_frame.grid(row=2, column=0, columnspan=3, padx=20, pady=20, sticky="ew")
        
        ctk.CTkLabel(algo_frame, text="🎯 Active Algorithm:", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=20, pady=20)
        
        self.algo_label = ctk.CTkLabel(algo_frame, text="CSP (Constraint Satisfaction)", 
                                      font=ctk.CTkFont(size=16))
        self.algo_label.pack(side="left", padx=10, pady=20)
        
        # Quick actions
        actions_frame = ctk.CTkFrame(tab)
        actions_frame.grid(row=3, column=0, columnspan=3, padx=20, pady=10, sticky="ew")
        
        ctk.CTkButton(actions_frame, text="🔄 Refresh Stats", 
                     command=self.update_stats).pack(side="left", padx=10, pady=10)
        ctk.CTkButton(actions_frame, text="📊 View Details", 
                     command=self.show_details).pack(side="left", padx=10, pady=10)

    def setup_tasks_tab(self):
        """Setup must-do tasks tab with date picker"""
        tab = self.tabview.tab("Must-Do Tasks")
        tab.grid_columnconfigure(0, weight=1)
        
        # Title
        title = ctk.CTkLabel(tab, text="📋 Must-Do Tasks Management", 
                            font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Input Frame
        input_frame = ctk.CTkFrame(tab)
        input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Task Name
        row = 0
        ctk.CTkLabel(input_frame, text="Task Name:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.task_name_entry = ctk.CTkEntry(input_frame, placeholder_text="e.g., Complete project report")
        self.task_name_entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        
        # Duration
        row += 1
        ctk.CTkLabel(input_frame, text="Duration (min):").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.task_duration_entry = ctk.CTkEntry(input_frame)
        self.task_duration_entry.insert(0, "30")
        self.task_duration_entry.grid(row=row, column=1, padx=10, pady=5, sticky="w")
        
        # Priority
        row += 1
        ctk.CTkLabel(input_frame, text="Priority (1-10):").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.task_priority_slider = ctk.CTkSlider(input_frame, from_=1, to=10, number_of_steps=9)
        self.task_priority_slider.set(5)
        self.task_priority_slider.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        self.priority_label = ctk.CTkLabel(input_frame, text="5")
        self.priority_label.grid(row=row, column=2, padx=5, pady=5)
        
        # Deadline
        row += 1
        ctk.CTkLabel(input_frame, text="Deadline (YYYY-MM-DD):").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        
        # Create date entry with today's date as default
        today = datetime.now().strftime("%Y-%m-%d")
        self.task_deadline_entry = ctk.CTkEntry(input_frame, placeholder_text=today)
        self.task_deadline_entry.insert(0, today)
        self.task_deadline_entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        
        # Emotions
        row += 1
        ctk.CTkLabel(input_frame, text="Best Emotions:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        
        emotions = ["sad", "happy", "neutral", "fear", "surprise", "disgust", "angry"]
        self.emotion_vars = {}
        emotion_frame = ctk.CTkFrame(input_frame)
        emotion_frame.grid(row=row, column=1, padx=10, pady=5, sticky="w")
        
        for i, emotion in enumerate(emotions):
            var = tk.BooleanVar(value=(emotion in ["neutral"]))
            cb = ctk.CTkCheckBox(emotion_frame, text=emotion.capitalize(), variable=var)
            cb.grid(row=0, column=i, padx=5)
            self.emotion_vars[emotion] = var
        
        # Add Task Button
        row += 1
        self.add_task_btn = ctk.CTkButton(input_frame, text="➕ Add Must-Do Task", 
                                         command=self.add_task, height=40)
        self.add_task_btn.grid(row=row, column=0, columnspan=3, padx=10, pady=15)
        
        # Bind priority slider update
        self.task_priority_slider.configure(command=self.update_priority_label)
        
        # Task List Frame
        list_frame = ctk.CTkFrame(tab)
        list_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        tab.grid_rowconfigure(2, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(list_frame, text="Current Must-Do Tasks:", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.task_list_scroll = ctk.CTkScrollableFrame(list_frame, height=300)
        self.task_list_scroll.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_rowconfigure(1, weight=1)
        
        self.task_list_container = self.task_list_scroll

    def update_priority_label(self, value):
        """Update priority label when slider moves"""
        self.priority_label.configure(text=str(int(float(value))))

    def add_task(self):
        """Add a new must-do task with deadline"""
        name = self.task_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a task name")
            return
        
        try:
            duration = int(self.task_duration_entry.get())
            priority = int(self.task_priority_slider.get())
            deadline = self.task_deadline_entry.get().strip()
            
            # Validate date format
            if deadline:
                try:
                    datetime.strptime(deadline, "%Y-%m-%d")
                except ValueError:
                    messagebox.showerror("Error", "Deadline must be in YYYY-MM-DD format")
                    return
            
            # Get selected emotions
            selected_emotions = [emotion for emotion, var in self.emotion_vars.items() if var.get()]
            if not selected_emotions:
                selected_emotions = ["neutral"]
            
            # Create task
            task = Task(
                name=name,
                base_priority=priority,
                category="todo_must",
                duration=duration,
                emotion_fit=selected_emotions,
                deadline=deadline if deadline else None,
                task_type="must_do"
            )
            
            self.todo_tasks.append(task)
            self.refresh_task_list()
            self.task_name_entry.delete(0, tk.END)
            self.update_stats()
            
            messagebox.showinfo("Success", f"Task '{name}' added successfully!")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def refresh_task_list(self):
        """Refresh the task list display"""
        if not hasattr(self, 'task_list_container'):
            return
            
        for widget in self.task_list_container.winfo_children():
            widget.destroy()
        
        if not self.todo_tasks:
            label = ctk.CTkLabel(self.task_list_container, text="No tasks added yet", 
                                text_color="gray", font=ctk.CTkFont(size=14))
            label.pack(pady=20)
            return
        
        for i, task in enumerate(self.todo_tasks):
            frame = ctk.CTkFrame(self.task_list_container)
            frame.pack(fill="x", padx=5, pady=5, expand=True)
            frame.grid_columnconfigure(0, weight=1)
            
            # Task info
            info_frame = ctk.CTkFrame(frame, fg_color="transparent")
            info_frame.grid(row=0, column=0, sticky="w", padx=10, pady=5)
            
            ctk.CTkLabel(info_frame, text=f"📝 {task.name}", 
                        font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
            
            details = f"Duration: {task.duration}min | Priority: {task.base_priority}"
            if hasattr(task, 'deadline') and task.deadline:
                details += f" | Deadline: {task.deadline}"
            ctk.CTkLabel(info_frame, text=details, text_color="gray").pack(anchor="w")
            
            # Emotions
            if hasattr(task, 'emotion_fit') and task.emotion_fit:
                emotions = ", ".join(task.emotion_fit)
                ctk.CTkLabel(info_frame, text=f"Emotions: {emotions}", 
                            text_color="lightblue").pack(anchor="w")
            
            # Remove button
            ctk.CTkButton(frame, text="🗑️", width=40, height=30, fg_color="transparent", 
                         text_color="red", hover_color="#3a3a3a",
                         command=lambda t=task: self.remove_task(t)).grid(row=0, column=1, padx=10)

    def remove_task(self, task):
        """Remove a task from the list"""
        self.todo_tasks.remove(task)
        self.refresh_task_list()
        self.update_stats()

    def setup_mood_tab(self):
        """Setup mood activities tab"""
        tab = self.tabview.tab("Mood Activities")
        tab.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(tab, text="😊 Mood-Changing Activities", 
                            font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Default activities
        default_frame = ctk.CTkFrame(tab)
        default_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(default_frame, text="Select Default Activities:", 
                    font=ctk.CTkFont(size=16)).pack(anchor="w", padx=10, pady=10)
        
        self.mood_scroll = ctk.CTkScrollableFrame(default_frame, height=200)
        self.mood_scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load default tasks
        default_tasks = get_default_mood_tasks()
        self.mood_vars = []
        
        for task in default_tasks:
            var = tk.BooleanVar(value=True)
            frame = ctk.CTkFrame(self.mood_scroll)
            frame.pack(fill="x", padx=5, pady=2)
            
            cb = ctk.CTkCheckBox(frame, text="", variable=var, width=20)
            cb.pack(side="left", padx=5)
            
            ctk.CTkLabel(frame, text=f"{task.name} ({task.duration}min)", 
                        font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
            
            if hasattr(task, 'emotion_fit') and task.emotion_fit:
                emotions = ", ".join(task.emotion_fit[:2])
                ctk.CTkLabel(frame, text=f"Helps: {emotions}", 
                            text_color="gray", font=ctk.CTkFont(size=10)).pack(side="left", padx=10)
            
            self.mood_vars.append((var, task))
        
        # Add all button
        ctk.CTkButton(default_frame, text="✅ Select All Default Activities", 
                     command=self.select_all_moods).pack(pady=10)

    def select_all_moods(self):
        """Select all mood activities"""
        for var, _ in self.mood_vars:
            var.set(True)
        self.update_stats()

    def setup_preferences_tab(self):
        """Setup preferences tab with date/time constraints"""
        tab = self.tabview.tab("Preferences")
        tab.grid_columnconfigure(0, weight=1)
        
        title = ctk.CTkLabel(tab, text="⚙️ CSP Preferences & Constraints", 
                            font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Input Frame
        input_frame = ctk.CTkFrame(tab)
        input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Activity Name
        row = 0
        ctk.CTkLabel(input_frame, text="Preference Name:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.pref_name_entry = ctk.CTkEntry(input_frame, placeholder_text="e.g., Meditate, Take a walk")
        self.pref_name_entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        
        # Time Range
        row += 1
        ctk.CTkLabel(input_frame, text="Time Range (24h format):").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        
        time_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        time_frame.grid(row=row, column=1, padx=10, pady=5, sticky="w")
        
        self.pref_start_hour = ctk.CTkEntry(time_frame, width=50, placeholder_text="9")
        self.pref_start_hour.insert(0, "9")
        self.pref_start_hour.pack(side="left")
        
        ctk.CTkLabel(time_frame, text=" to ").pack(side="left", padx=5)
        
        self.pref_end_hour = ctk.CTkEntry(time_frame, width=50, placeholder_text="17")
        self.pref_end_hour.insert(0, "17")
        self.pref_end_hour.pack(side="left")
        
        ctk.CTkLabel(time_frame, text=" hours").pack(side="left", padx=5)
        
        # Resources
        row += 1
        ctk.CTkLabel(input_frame, text="Required Resources:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        self.pref_resources = ctk.CTkEntry(input_frame, placeholder_text="comma-separated, e.g., headphones,outdoors")
        self.pref_resources.grid(row=row, column=1, padx=10, pady=5, sticky="ew")
        
        # Emotions
        row += 1
        ctk.CTkLabel(input_frame, text="Helps When Feeling:").grid(row=row, column=0, padx=10, pady=5, sticky="w")
        
        emotions = ["sad", "happy", "neutral", "fear", "surprise", "disgust", "angry"]
        self.pref_emotion_vars = {}
        emotion_frame = ctk.CTkFrame(input_frame)
        emotion_frame.grid(row=row, column=1, padx=10, pady=5, sticky="w")
        
        for i, emotion in enumerate(emotions):
            var = tk.BooleanVar(value=(emotion in ["sad", "angry", "fear"]))
            cb = ctk.CTkCheckBox(emotion_frame, text=emotion.capitalize(), variable=var)
            cb.grid(row=0, column=i, padx=5)
            self.pref_emotion_vars[emotion] = var
        
        # Add button
        row += 1
        self.add_pref_btn = ctk.CTkButton(input_frame, text="🎯 Add CSP Preference", 
                                         command=self.add_preference, height=40)
        self.add_pref_btn.grid(row=row, column=0, columnspan=2, padx=10, pady=15)
        
        # Preferences List
        list_frame = ctk.CTkFrame(tab)
        list_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
        tab.grid_rowconfigure(2, weight=1)
        
        ctk.CTkLabel(list_frame, text="Current Preferences:", 
                    font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.pref_list_scroll = ctk.CTkScrollableFrame(list_frame, height=250)
        self.pref_list_scroll.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_rowconfigure(1, weight=1)
        
        self.pref_list_container = self.pref_list_scroll

    def add_preference(self):
        """Add a new preference with time constraints"""
        name = self.pref_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a preference name")
            return
        
        try:
            # Get time constraints
            start_hour = int(self.pref_start_hour.get())
            end_hour = int(self.pref_end_hour.get())
            
            if not (0 <= start_hour < 24 and 0 <= end_hour <= 24):
                messagebox.showerror("Error", "Hours must be between 0-24")
                return
            
            # Get resources
            resources_text = self.pref_resources.get().strip()
            resources = [r.strip() for r in resources_text.split(',')] if resources_text else []
            
            # Get emotions
            selected_emotions = [emotion for emotion, var in self.pref_emotion_vars.items() if var.get()]
            if not selected_emotions:
                selected_emotions = ["neutral"]
            
            # Create preference task with constraints
            constraints = {
                "allowed_time": {"start": start_hour, "end": end_hour},
                "requires": resources
            }
            
            pref_task = Task(
                name=name,
                base_priority=8, 
                category="user_preference",
                duration=30,
                emotion_fit=selected_emotions,
                task_type="mood_changer",
                constraints=constraints
            )
            pref_task.is_preference = True
            
            self.csp_preferences.append(pref_task)
            self.refresh_pref_list()
            self.pref_name_entry.delete(0, tk.END)
            self.update_stats()
            
            messagebox.showinfo("Success", f"Preference '{name}' added!")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for hours")

    def refresh_pref_list(self):
        """Refresh the preferences list"""
        if not hasattr(self, 'pref_list_container'):
            return
            
        for widget in self.pref_list_container.winfo_children():
            widget.destroy()
        
        if not self.csp_preferences:
            label = ctk.CTkLabel(self.pref_list_container, text="No preferences added yet", 
                                text_color="gray", font=ctk.CTkFont(size=14))
            label.pack(pady=20)
            return
        
        for pref in self.csp_preferences:
            frame = ctk.CTkFrame(self.pref_list_container)
            frame.pack(fill="x", padx=5, pady=5, expand=True)
            frame.grid_columnconfigure(0, weight=1)
            
            # Preference info
            info_frame = ctk.CTkFrame(frame, fg_color="transparent")
            info_frame.grid(row=0, column=0, sticky="w", padx=10, pady=5)
            
            ctk.CTkLabel(info_frame, text=f"🎯 {pref.name}", 
                        font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
            
            # Show emotions
            if hasattr(pref, 'emotion_fit') and pref.emotion_fit:
                emotions_str = ", ".join(pref.emotion_fit)
                ctk.CTkLabel(info_frame, text=f"Helps when feeling: {emotions_str}", 
                           text_color="yellow").pack(anchor="w")
            
            # Show constraints
            if hasattr(pref, 'constraints') and pref.constraints:
                constraints = pref.constraints
                if "allowed_time" in constraints:
                    time_info = constraints["allowed_time"]
                    time_str = f"Time: {time_info.get('start', '?')}:00-{time_info.get('end', '?')}:00"
                    ctk.CTkLabel(info_frame, text=time_str, text_color="lightblue").pack(anchor="w")
                
                if "requires" in constraints and constraints["requires"]:
                    resources = ", ".join(constraints["requires"]) if isinstance(constraints["requires"], list) else constraints["requires"]
                    ctk.CTkLabel(info_frame, text=f"Resources: {resources}", text_color="lightgreen").pack(anchor="w")
            
            # Remove button
            ctk.CTkButton(frame, text="🗑️", width=40, height=30, fg_color="transparent", 
                         text_color="red", hover_color="#3a3a3a",
                         command=lambda p=pref: self.remove_pref(p)).grid(row=0, column=1, padx=10)

    def remove_pref(self, pref):
        """Remove a preference"""
        self.csp_preferences.remove(pref)
        self.refresh_pref_list()
        self.update_stats()

    def setup_camera_tab(self):
        """Setup camera tab with live feed"""
        tab = self.tabview.tab("Live Camera")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        
        # Title
        title = ctk.CTkLabel(tab, text="📹 Live Emotion Detection", 
                            font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Camera container
        camera_container = ctk.CTkFrame(tab)
        camera_container.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        camera_container.grid_columnconfigure(0, weight=1)
        camera_container.grid_rowconfigure(1, weight=1)
        
        # Controls
        controls_frame = ctk.CTkFrame(camera_container)
        controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.camera_btn = ctk.CTkButton(controls_frame, text="▶️ Start Camera", 
                                       command=self.toggle_camera, width=150)
        self.camera_btn.pack(side="left", padx=10, pady=10)
        
        self.emotion_label = ctk.CTkLabel(controls_frame, text="Emotion: neutral", 
                                         font=ctk.CTkFont(size=16, weight="bold"))
        self.emotion_label.pack(side="left", padx=20, pady=10)
        
        # Camera display
        self.camera_frame = ctk.CTkFrame(camera_container)
        self.camera_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Camera feed will appear here")
        self.camera_label.pack(expand=True, padx=20, pady=20)
        
        # Stats
        stats_frame = ctk.CTkFrame(tab)
        stats_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.camera_stats = ctk.CTkLabel(stats_frame, text="FPS: -- | Detections: 0", 
                                        text_color="gray")
        self.camera_stats.pack(pady=10)

    def setup_start_tab(self):
        """Setup session start tab"""
        tab = self.tabview.tab("Start Session")
        tab.grid_columnconfigure(0, weight=1)
        
        # Title
        title = ctk.CTkLabel(tab, text="🚀 Start Optimization Session", 
                            font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, padx=20, pady=(30, 20), sticky="w")
        
        # Algorithm selection
        algo_frame = ctk.CTkFrame(tab)
        algo_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(algo_frame, text="Select Algorithm:", 
                    font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))
        
        # Algorithm options
        algos = [
            ("🔮 Mini A*", "Finds optimal sequence considering emotional transitions", 1),
            ("📈 Greedy", "Always picks highest scoring task", 2),
            ("⚖️ CSP (Recommended)", "Balanced with strict time/resource constraints", 3),
            ("🎲 Stochastic", "Random selection weighted by score", 4),
            ("⛰️ Hill Climbing", "Step-by-step optimization", 5)
        ]
        
        for name, desc, value in algos:
            frame = ctk.CTkFrame(algo_frame, fg_color="transparent")
            frame.pack(fill="x", padx=20, pady=5)
            
            rb = ctk.CTkRadioButton(frame, text=name, variable=self.algorithm_choice, 
                                   value=value, font=ctk.CTkFont(size=14),
                                   command=self.update_recommendations_panel)
            rb.pack(side="left")
            
            ctk.CTkLabel(frame, text=desc, text_color="gray", 
                        font=ctk.CTkFont(size=12)).pack(side="left", padx=20)
        
        # Session settings
        settings_frame = ctk.CTkFrame(tab)
        settings_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        
        ctk.CTkLabel(settings_frame, text="Session Settings:", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=20, pady=(20, 10))
        
        # Deadline warnings checkbox
        deadline_cb = ctk.CTkCheckBox(settings_frame, text="Show deadline warnings", 
                                     variable=self.use_deadline_warnings)
        deadline_cb.pack(anchor="w", padx=20, pady=10)
        
        # Resources input
        resources_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        resources_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(resources_frame, text="Available Resources:").pack(side="left", padx=(0, 10))
        self.resources_entry = ctk.CTkEntry(resources_frame, width=300, 
                                           placeholder_text="computer, internet, headphones, notebook")
        self.resources_entry.insert(0, "computer, internet, water")
        self.resources_entry.pack(side="left")
        
        # Start button
        self.start_btn = ctk.CTkButton(tab, text="🚀 Launch Integrated Session", 
                                      height=60, font=ctk.CTkFont(size=18, weight="bold"),
                                      command=self.start_gui_session)
        self.start_btn.grid(row=3, column=0, padx=20, pady=40, sticky="ew")
        
        # Status
        self.session_status = ctk.CTkLabel(tab, text="Status: Ready to start session", 
                                          text_color="gray", font=ctk.CTkFont(size=14))
        self.session_status.grid(row=4, column=0, padx=20, pady=10)
        
        self.start_emotion_label = ctk.CTkLabel(tab, text="Current Emotion: (No Camera)", 
                                               font=ctk.CTkFont(size=16, weight="bold"))
        self.start_emotion_label.grid(row=5, column=0, padx=20, pady=10)

    def update_stats(self):
        """Update statistics display"""
        if not hasattr(self, 'stats_widgets'):
            return
            
        # Update task counts
        if 'Must-Do Tasks' in self.stats_widgets:
            self.stats_widgets['Must-Do Tasks'].configure(text=str(len(self.todo_tasks)))
        
        if 'Mood Activities' in self.stats_widgets:
            selected_count = sum(1 for var, _ in getattr(self, 'mood_vars', [])) if hasattr(self, 'mood_vars') else 0
            self.stats_widgets['Mood Activities'].configure(text=str(selected_count))
        
        if 'Preferences' in self.stats_widgets:
            self.stats_widgets['Preferences'].configure(text=str(len(self.csp_preferences)))
        
        if 'Current Emotion' in self.stats_widgets:
            self.stats_widgets['Current Emotion'].configure(text=self.current_emotion)
        
        # Update algorithm label
        if hasattr(self, 'algo_label'):
            algo_map = {
                1: "Mini A*",
                2: "Greedy", 
                3: "CSP (Constraint Satisfaction)",
                4: "Stochastic",
                5: "Hill Climbing"
            }
            self.algo_label.configure(text=algo_map.get(self.algorithm_choice.get(), "CSP"))

    def setup_active_session_tab(self):
        """Setup the Active Session interface"""
        tab = self.tabview.tab("Active Session")
        tab.grid_columnconfigure(0, weight=1)
        
        # Header
        self.session_title = ctk.CTkLabel(tab, text="⏱️ No Active Session", 
                                         font=ctk.CTkFont(size=24, weight="bold"))
        self.session_title.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        # Timer - LARGE display
        self.timer_label = ctk.CTkLabel(tab, text="00:00:00", 
                                       font=ctk.CTkFont(size=72, weight="bold"),
                                       text_color="#1abc9c")
        self.timer_label.grid(row=1, column=0, padx=20, pady=30)
        
        # Session Camera Feed (Secondary)
        self.session_camera_label = ctk.CTkLabel(tab, text="")
        self.session_camera_label.grid(row=2, column=0, padx=20, pady=5)
        
        self.session_emotion_label = ctk.CTkLabel(tab, text="Feeling: NEUTRAL", 
                                                font=ctk.CTkFont(size=20, weight="bold"),
                                                text_color="#1abc9c")
        self.session_emotion_label.grid(row=3, column=0, padx=20, pady=5)
        
        # Current Task Info
        self.current_task_frame = ctk.CTkFrame(tab)
        self.current_task_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        self.active_task_name = ctk.CTkLabel(self.current_task_frame, text="Current Task: None",
                                            font=ctk.CTkFont(size=18))
        self.active_task_name.pack(pady=10)
        
        self.active_task_details = ctk.CTkLabel(self.current_task_frame, text="Duration: -- | Started: --",
                                               text_color="gray")
        self.active_task_details.pack(pady=5)
        
        # Controls
        self.session_controls = ctk.CTkFrame(tab, fg_color="transparent")
        self.session_controls.grid(row=5, column=0, padx=20, pady=10)
        
        self.finish_btn = ctk.CTkButton(self.session_controls, text="✅ Finish Task", 
                                       command=self.complete_current_task,
                                       height=40, width=150)
        self.finish_btn.pack(side="left", padx=10)
        
        self.stop_session_btn = ctk.CTkButton(self.session_controls, text="🛑 Quit Session", 
                                             fg_color="#e74c3c", hover_color="#c0392b",
                                             command=self.stop_gui_session,
                                             height=40, width=150)
        self.stop_session_btn.pack(side="left", padx=10)

    def stop_gui_session(self):
        """End the GUI session"""
        self.is_session_active = False
        
        # Stop all timers
        self.stop_todo_timer()
        self.stop_break_timer()
        
        # Reset timer states
        self.todo_timer_running = False
        self.break_timer_running = False
        
        # Update UI
        if hasattr(self, 'session_title'):
            self.session_title.configure(text="⏱️ No Active Session")
        if hasattr(self, 'timer_label'):
            self.timer_label.configure(text="00:00:00", text_color="gray")
        if hasattr(self, 'active_task_name'):
            self.active_task_name.configure(text="Current Task: None")
        if hasattr(self, 'sidebar_status'):
            self.sidebar_status.configure(text="Status: Ready", text_color="gray")
        
        self.tabview.set("Dashboard")
        self.stop_camera()
        
        # Clear recommendations
        self.update_recommendations_panel()

    def show_details(self):
        """Show detailed statistics"""
        details = f"""
        📊 Session Details:
        
        Must-Do Tasks: {len(self.todo_tasks)}
        Mood Activities: {sum(1 for var, _ in getattr(self, 'mood_vars', [])) if hasattr(self, 'mood_vars') else 0}
        CSP Preferences: {len(self.csp_preferences)}
        
        Tasks with Deadlines: {sum(1 for t in self.todo_tasks if hasattr(t, 'deadline') and t.deadline)}
        Preferences with Time Constraints: {len(self.csp_preferences)}
        
        Current Algorithm: {self.algo_label.cget('text') if hasattr(self, 'algo_label') else "CSP"}
        Current Emotion: {self.current_emotion}
        """
        messagebox.showinfo("Session Details", details)

    def run_session_loop(self):
        """Regularly update the session state"""
        if not self.is_session_active:
            return
            
        self.after(1000, self.run_session_loop)

    def load_saved_data(self):
        """Load saved data from previous session"""
        if os.path.exists("last_session.pkl"):
            try:
                with open("last_session.pkl", 'rb') as f:
                    data = pickle.load(f)
                
                self.todo_tasks = data.get('todo_tasks', [])
                self.csp_preferences = data.get('csp_preferences', [])
                
                if hasattr(self, 'refresh_task_list'):
                    self.refresh_task_list()
                if hasattr(self, 'refresh_pref_list'):
                    self.refresh_pref_list()
                
                messagebox.showinfo("Info", "Previous session data loaded successfully!")
            except Exception as e:
                print(f"Error loading saved data: {e}")

    def on_closing(self):
        """Handle window closing"""
        self.monitoring = False
        
        # Stop all timers
        self.stop_todo_timer()
        self.stop_break_timer()
        
        if self.camera_running:
            self.stop_camera()
        
        # Save current session
        try:
            data = {
                'todo_tasks': self.todo_tasks,
                'csp_preferences': self.csp_preferences,
            }
            with open("last_session.pkl", 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving session: {e}")
        
        self.destroy()


if __name__ == "__main__":
    app = TaskOptimizerGUI()
    app.mainloop()