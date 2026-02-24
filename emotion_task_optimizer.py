import time
import os
import cv2
import pickle
from datetime import datetime
from task import Task
from algorithms import csp_filter, greedy, hill_climbing, stochastic, mini_a_star, multi_objective_optimization, analyze_csp_failure
from algorithms import csp_task, csp_preferences, create_preference_task
from emotion_detector import start_camera
from default_tasks import get_default_mood_tasks

SUPPORTED_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class TaskManager:
    """
    Manages THREE separate task categories:
    1. To-Do List (mandatory tasks)
    2. Mood-Changing Activities (what user likes to do)
    3. Default Fallback Activities
    4. Custom Preferences for CSP
    """
    def __init__(self):
        self.todo_tasks = []      # Must complete
        self.mood_activities = [] # Like to do for mood change
        self.user_preferences = [] # Custom preferences for CSP_preferences
    
    def load_todo_list(self):
        """Load user's mandatory to-do tasks"""
        print("\n" + "="*60)
        print("📋 STEP 1: Your MUST-DO TASKS")
        print("These are tasks you NEED to complete (work, study, chores)")
        print("="*60 + "\n")
        
        while True:
            print(f"Task #{len(self.todo_tasks) + 1}")
            name = input("Task name (or 'done' to finish): ")
            if name.lower() == "done":
                break
            
            # Priority based on urgency/importance
            print("\nHow important/urgent is this task?")
            print("1. Critical (do today)")
            print("2. High priority (do this week)")
            print("3. Medium priority (when possible)")
            print("4. Low priority (backlog)")
            
            priority_map = {"1": 9, "2": 7, "3": 5, "4": 3}
            while True:
                choice = input("Choice (1-4): ")
                if choice in priority_map:
                    base_priority = priority_map[choice]
                    break
                print("Please enter 1-4")
            
            duration = int(input(f"Estimated time needed (minutes): "))
            
            # Emotional suitability - which emotions make this task doable?
            print("\nWhich emotional states are BEST for this task?")
            print("Example: 'neutral,focused' for studying")
            print("Emotions: " + ", ".join(SUPPORTED_EMOTIONS))
            emotion_input = input("Enter emotions (comma-separated, Enter for 'neutral'): ")
            
            if emotion_input.strip():
                emotion_fit = [e.strip().lower() for e in emotion_input.split(",")]
                emotion_fit = [e for e in emotion_fit if e in SUPPORTED_EMOTIONS]
            else:
                emotion_fit = ["neutral"]
            
            # Deadline (optional)
            deadline = input("Deadline (YYYY-MM-DD or Enter for none): ")
            
            task = Task(
                name=name,
                base_priority=base_priority,
                category="todo_must",
                duration=duration,
                emotion_fit=emotion_fit,
                deadline=deadline if deadline else None,
                task_type="must_do"
            )
            
            self.todo_tasks.append(task)
            print(f"✓ Added MUST-DO: {name} (Priority: {base_priority}, Duration: {duration}min)\n")
        
        # If no tasks, add some defaults
        if not self.todo_tasks:
            print("\n⚠ No must-do tasks added. Adding sample tasks...")
            sample_task = Task(
                name="Review your goals for the day",
                base_priority=5,
                category="planning",
                duration=10,
                emotion_fit=["neutral", "calm"],
                task_type="must_do"
            )
            self.todo_tasks.append(sample_task)
        
        return self.todo_tasks
    
    def load_mood_activities(self, ask_for_custom=True):
        """Load activities user LIKES to do to change their mood"""
        print("\n" + "="*60)
        print("😊 STEP 2: Your MOOD-CHANGING ACTIVITIES")
        print("What do you ENJOY doing to improve your mood?")
        print("These are OPTIONAL activities for when you need a break")
        print("="*60 + "\n")
        
        # First, load default tasks as fallback
        default_tasks = get_default_mood_tasks()
        
        # Show default tasks
        print("\n📋 DEFAULT MOOD-CHANGING ACTIVITIES")
        print("Select which ones you'd like to include:")
        
        for i, task in enumerate(default_tasks, 1):
            print(f"{i}. {task.name} ({task.duration}min) - Helps with: {', '.join(task.emotion_fit[:3])}")
        
        selected = input("\nSelect numbers (comma-separated), 'all' for all, or Enter to skip: ").strip().lower()
        
        if selected == 'all':
            self.mood_activities.extend(default_tasks)
            print(f"✓ Added all {len(default_tasks)} default activities")
        elif selected:
            indices = []
            for part in selected.split(','):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(default_tasks):
                        indices.append(idx)
            
            for idx in indices:
                task = default_tasks[idx]
                self.mood_activities.append(task)
                print(f"✓ Added: {task.name}")
        
        # Only ask for custom activities if enabled
        if ask_for_custom:
            print("\nAdd your OWN mood-changing activities:")
            while True:
                name = input("\nActivity name (or 'done' to finish): ")
                if name.lower() == "done":
                    break
                
                # Which emotions does this help with?
                print("\nWhen you feel ____, doing this helps:")
                print("Available: " + ", ".join(SUPPORTED_EMOTIONS))
                emotion_input = input("Enter emotions (comma-separated): ")
                emotion_fit = [e.strip().lower() for e in emotion_input.split(",")]
                emotion_fit = [e for e in emotion_fit if e in SUPPORTED_EMOTIONS]
                
                if not emotion_fit:
                    emotion_fit = ["sad", "angry"]  # Default
                
                duration = int(input("How long does this activity take (minutes)? "))
                
                # How effective is it?
                effectiveness = int(input("How effective is this for mood change? (1-10, 10=very): "))
                base_priority = max(1, min(10, effectiveness))
                
                constraints = {}
                requires = input("What do you need? (e.g., headphones, outdoors, Enter for none): ")
                if requires:
                    constraints["requires"] = requires
                
                task = Task(
                    name=name,
                    base_priority=base_priority,
                    category="mood_enhancer",
                    duration=duration,
                    emotion_fit=emotion_fit,
                    task_type="mood_changer",
                    constraints=constraints
                )
                
                self.mood_activities.append(task)
                print(f"✓ Added: {name} (Effectiveness: {effectiveness}/10)\n")
        
        print(f"\n✓ Total mood activities: {len(self.mood_activities)}")
        return self.mood_activities
    
    def load_emotion_task_preferences(self):
        """Map which MUST-DO tasks are suitable for which emotions"""
        preferences = {}
        if not self.todo_tasks:
            return preferences
        
        for emotion in SUPPORTED_EMOTIONS:
            suitable_tasks = [t for t in self.todo_tasks if emotion in t.emotion_fit]
            if suitable_tasks:
                preferences[emotion] = suitable_tasks
        return preferences
    
    def load_user_preferences_csp(self):
        """Load user's custom preferences for CSP_preferences function"""
        self.user_preferences = []
        while True:
            name = input("\nPreference Activity name (or 'done' to finish): ")
            if name.lower() == "done":
                break
            
            print("\nEmotions: " + ", ".join(SUPPORTED_EMOTIONS))
            emotion_input = input("Enter emotions (comma-separated): ")
            emotions = [e.strip().lower() for e in emotion_input.split(",")]
            emotions = [e for e in emotions if e in SUPPORTED_EMOTIONS]
            
            print("\nAllowed time (start-end, 24h format, e.g., 9-17):")
            time_input = input("Time (or Enter for any): ")
            time_constraint = None
            if time_input:
                try:
                    s, e = time_input.split("-")
                    time_constraint = {"start": int(s), "end": int(e)}
                except:
                    pass
            
            resources_input = input("\nResources (comma-separated): ")
            resources = [r.strip().lower() for r in resources_input.split(",")] if resources_input.strip() else []
            
            effectiveness = int(input("Effectiveness (1-10): "))
            duration = int(input("Duration (minutes): "))
            
            task = create_preference_task(name, emotions, time_constraint, resources, effectiveness)
            task.duration = duration
            self.user_preferences.append(task)
        
        return self.user_preferences

def check_time_constraints(pref_task, current_time):
    current_hour = current_time.hour
    if "allowed_time" in pref_task.constraints:
        time_info = pref_task.constraints["allowed_time"]
        if isinstance(time_info, dict):
            allowed_start = time_info.get("start", 0)
            allowed_end = time_info.get("end", 24)
            if allowed_start <= current_hour < allowed_end:
                 return True, "✅ Within allowed time range."
            return False, f"❌ Outside allowed range ({allowed_start}:00-{allowed_end}:00)."
    return True, "✅ No time constraints."

def dynamic_resource_check(pref_task, current_conditions):
    if "requires" in pref_task.constraints:
        required = pref_task.constraints["requires"]
        required = [required] if isinstance(required, str) else required
        available = current_conditions.get("available_resources", [])
        missing = [r for r in required if r not in available]
        if missing:
            print(f"   Missing: {', '.join(missing)}")
            if input(f"   Have {', '.join(missing)}? (y/n): ").lower() == 'y':
                 available.extend(missing)
                 return True
            return False
    return True

def check_preference_tasks_for_emotion(emotion, user_preferences, current_conditions, completed_tasks):
    available = []
    for task in user_preferences:
        if emotion in task.emotion_fit and task not in completed_tasks:
            if dynamic_resource_check(task, current_conditions):
                if check_time_constraints(task, current_conditions["current_time"])[0]:
                    task.compute_score(emotion, current_conditions.get("current_energy", 5))
                    if getattr(task, 'is_preference', False):
                        task.score += 30.0
                    available.append(task)
    available.sort(key=lambda x: x.score, reverse=True)
    return available

def draw_recommendations(frame, tasks, title="Recommended Activities"):
    y = 140
    cv2.putText(frame, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    y += 30
    for i, task in enumerate(tasks[:3], 1):
        cv2.putText(
            frame,
            f"{i}. {task.name} ({task.duration} min)",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1
        )
        y += 30
def draw_recommendation_panel(frame, recommended_tasks):
    h, w, _ = frame.shape
    panel_width = 300
    x_start = w - panel_width

    # Panel background
    cv2.rectangle(frame,
                  (x_start, 0),
                  (w, h),
                  (40, 40, 40),
                  -1)

    # Title
    cv2.putText(frame,
                "Mood Recommendations",
                (x_start + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)

    # Tasks
    if not recommended_tasks:
        cv2.putText(frame,
                    "No suggestions",
                    (x_start + 10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1)
        return

    for i, task in enumerate(recommended_tasks[:5]):
        y = 70 + i * 35
        cv2.putText(frame,
                    f"- {task.name}",
                    (x_start + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1)

def integrated_task_mode(todo_tasks, mood_activities, emotion_preferences,
                         user_preferences, selected_algorithm,
                         use_deadline_csp=True, use_preference_csp=True):

    cap, detector = start_camera()

    # ================= STATE =================
    ui_state = "TODO_SELECTION"     # TODO_SELECTION | WORKING_TODO | BREAK
    recommended_tasks = []
    interrupted_task = None
    current_task = None
    completed_tasks = []

    task_start_time = None
    remaining_time = 0

    last_emotion = "neutral"
    emotion_detection_paused = False

    current_conditions = {
        "current_time": datetime.now(),
        "available_resources": ["computer", "internet"],
        "current_energy": 5,
        "current_emotion": "neutral"
    }

    print(f"\n🚀 SYSTEM ACTIVE - ALGO {selected_algorithm}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            current_conditions["current_time"] = datetime.now()

            # ======================================================
            # 🎭 EMOTION DETECTION → SEARCH ALGORITHMS
            # ======================================================
            if not emotion_detection_paused:
                emotion = detector.detect_emotion_from_frame(frame)

                if emotion != last_emotion:
                    last_emotion = emotion
                    current_conditions["current_emotion"] = emotion

                    print(f"🔍 Searching... Emotion: {emotion}")

                    pool = mood_activities + user_preferences
                    for t in pool:
                        t.is_preference = True

                    # ---------- RUN SEARCH ----------
                    if selected_algorithm == 3:  # CSP
                        recommended_tasks, _ = csp_preferences(pool, current_conditions)
                    elif selected_algorithm == 1:
                        recommended_tasks = mini_a_star([], emotion, 0, 3, pool)
                    elif selected_algorithm == 2:
                        best = greedy(pool, emotion, current_conditions)
                        recommended_tasks = [best] if best else []
                    elif selected_algorithm == 4:
                        best = stochastic(pool, emotion)
                        recommended_tasks = [best] if best else []
                    elif selected_algorithm == 5:
                        best = hill_climbing(pool, current_conditions)
                        recommended_tasks = [best] if best else []
                    else:
                        recommended_tasks = []

                    # ---------- 🔥 AUTO-SHOW BREAK UI ----------
                    if recommended_tasks and ui_state == "WORKING_TODO":
                        interrupted_task = current_task
                        current_task = None
                        ui_state = "BREAK"
                        emotion_detection_paused = True


            # ======================================================
            # 📋 TODO SELECTION
            # ======================================================
            if ui_state == "TODO_SELECTION":
                undone = [t for t in todo_tasks if t not in completed_tasks]

                if undone:
                    undone.sort(
                        key=lambda t: (
                            t.deadline if t.deadline else "9999-12-31",
                            -t.base_priority
                        )
                    )

                    cv2.putText(display,
                                f"Next TODO: {undone[0].name}",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2)

                    cv2.putText(display,
                                "S: Start  |  Q: Quit",
                                (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (200, 200, 200),
                                1)

            # ======================================================
            # ⏳ WORKING ON TODO (COUNTDOWN)
            # ======================================================
            if ui_state == "WORKING_TODO" and current_task:
                elapsed = time.time() - task_start_time
                remaining_time = max(0, current_task.duration * 60 - elapsed)

                m, s = divmod(int(remaining_time), 60)

                cv2.putText(display,
                            f"⏳ {m:02d}:{s:02d}",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)

                cv2.putText(display,
                            "B: Take Break",
                            (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1)

                if remaining_time <= 0:
                    completed_tasks.append(current_task)
                    print(f"✅ Completed TODO: {current_task.name}")
                    current_task = None
                    ui_state = "TODO_SELECTION"

            # ======================================================
            # 🧘 BREAK MODE
            # ======================================================
            if ui_state == "BREAK":
                cv2.putText(display,
                            "Break Suggestions (1–3):",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)

                for i, task in enumerate(recommended_tasks[:3], 1):
                    cv2.putText(display,
                                f"{i}. {task.name} ({task.duration}m)",
                                (10, 50 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),
                                1)

            # ======================================================
            # 🖥️ DISPLAY
            # ======================================================
            cv2.putText(display,
                        f"Emotion: {last_emotion.upper()}",
                        (10, display.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

            draw_recommendation_panel(display, recommended_tasks)
            cv2.imshow("Emotion Task Optimizer", display)

            # ======================================================
            # ⌨️ KEY HANDLING
            # ======================================================
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if ui_state == "TODO_SELECTION" and key == ord('s'):
                undone = [t for t in todo_tasks if t not in completed_tasks]
                if undone:
                    undone.sort(
                        key=lambda t: (
                            t.deadline if t.deadline else "9999-12-31",
                            -t.base_priority
                        )
                    )
                    current_task = undone[0]
                    task_start_time = time.time()
                    ui_state = "WORKING_TODO"
                    print(f"🚀 Started TODO: {current_task.name}")

            if ui_state == "WORKING_TODO" and key == ord('b'):
                interrupted_task = current_task
                ui_state = "BREAK"

            if ui_state == "BREAK" and key in [ord('1'), ord('2'), ord('3')]:
                idx = key - ord('1')
                if idx < len(recommended_tasks):
                    current_task = recommended_tasks[idx]
                    task_start_time = time.time()
                    emotion_detection_paused = True
                    print(f"🧘 Break: {current_task.name}")

            if ui_state == "BREAK" and current_task:
                elapsed = time.time() - task_start_time
                if elapsed >= current_task.duration * 60:
                    print("↩ Returning to TODO")
                    current_task = interrupted_task
                    interrupted_task = None
                    task_start_time = time.time()
                    ui_state = "WORKING_TODO"
                    emotion_detection_paused = False

    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Emotion Task Optimizer")
    parser.add_argument("--data", type=str, help="Path to session data pickle")
    parser.add_argument("--skip-prompts", action="store_true", help="Skip initial interactive prompts")
    args = parser.parse_args()

    manager = TaskManager()
    
    if args.data and os.path.exists(args.data):
        try:
            with open(args.data, 'rb') as f:
                data = pickle.load(f)
            
            todo_tasks = data.get('todo_tasks', [])
            mood_activities = data.get('mood_activities', [])
            user_preferences = data.get('csp_preferences', [])
            algorithm_choice = data.get('algorithm_choice', 3)
            use_deadline_csp = data.get('use_deadline_warnings', True)
            use_preference_csp = True 
            
            manager.todo_tasks = todo_tasks
            emotion_preferences = manager.load_emotion_task_preferences()
            
            print("\n" + "="*60)
            print("🚀 SESSION LOADED FROM GUI")
            print("="*60)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print("🎭 EMOTION TASK OPTIMIZER")
        choice = input("Algorithm (1-CSP): ")
        algorithm_choice = int(choice) if choice.isdigit() else 3
        todo_tasks = manager.load_todo_list()
        mood_activities = manager.load_mood_activities()
        user_preferences = manager.load_user_preferences_csp() if algorithm_choice == 3 else []
        emotion_preferences = manager.load_emotion_task_preferences()
        use_deadline_csp = True

    if not args.skip_prompts:
        input("\nPress Enter to start...")
    
    integrated_task_mode(todo_tasks, mood_activities, emotion_preferences, user_preferences, algorithm_choice, use_deadline_csp)

if __name__ == "__main__":
    main()