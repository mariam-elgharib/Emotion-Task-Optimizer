from datetime import datetime, timedelta

class Task:
    def __init__(self, name, base_priority, category, duration, emotion_fit, 
                 deadline=None, task_type="general", conditions=None, constraints=None):
        """
        task_type can be:
        - "must_do": User's mandatory tasks
        - "mood_changer": Activities to improve mood
        - "default": System fallback
        """
        self.name = name
        self.base_priority = base_priority
        self.category = category
        self.duration = duration
        self.emotion_fit = emotion_fit if isinstance(emotion_fit, list) else [emotion_fit]
        self.deadline = deadline
        self.task_type = task_type
        self.score = 0
        self.start_time = None
        self.created_at = datetime.now()
        
        # Additional attributes for enhanced searches
        self.conditions = conditions if conditions else []
        self.constraints = constraints if constraints else {}
        self.completed = False
        self.attempt_count = 0
        self.last_attempted = None
        self.success_rate = 1.0  # Start with 100% success rate
        self.elapsed_time = 0    # Track time spent on task across sessions
        
        # Time-based constraints
        self.time_constraints = self.constraints.get("time_constraints", {})
        self.preferred_time = self.constraints.get("preferred_time", "any")
        self.energy_required = self.constraints.get("energy_required", 5)  # 1-10 scale
        
    def compute_score(self, current_emotion, current_time=None, current_energy=5, 
                     preference_bonus=0, urgency_bonus=0):
        """
        Compute dynamic score based on:
        1. Base priority (importance)
        2. Emotion match
        3. Task type bonus
        4. Time suitability
        5. Energy level match
        6. Success history
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Emotion match bonus
        if current_emotion in self.emotion_fit:
            emotion_bonus = 8  # High bonus for emotional match
        else:
            emotion_bonus = -4  # Penalty for mismatch
        
        # Task type bonus
        if self.task_type == "must_do":
            type_bonus = 5
        elif self.task_type == "mood_changer":
            type_bonus = 3
        else:
            type_bonus = 0
        
        # Time suitability bonus
        time_bonus = self.get_time_suitability(current_time)
        
        # Energy match bonus
        energy_diff = abs(self.energy_required - current_energy)
        energy_bonus = max(0, 5 - energy_diff)  # Higher bonus when energy matches
        
        # Success rate bonus
        success_bonus = self.success_rate * 3
        
        # Urgency bonus (if deadline exists)
        urgency = self.get_urgency_bonus()
        
        self.score = (self.base_priority + 
                     emotion_bonus + 
                     type_bonus + 
                     time_bonus +
                     energy_bonus +
                     success_bonus +
                     preference_bonus + 
                     urgency)
        
        # Ensure score is positive
        self.score = max(0.1, self.score)
        
        return self.score
    
    def get_time_suitability(self, current_time):
        """Check if current time is suitable for this task"""
        hour = current_time.hour
        
        if "time_constraints" in self.constraints:
            time_constraints = self.constraints["time_constraints"]
            if "morning_only" in time_constraints and hour >= 12:
                return -5  # Not suitable
            if "evening_only" in time_constraints and hour < 17:
                return -5
            if "office_hours" in time_constraints and (hour < 9 or hour >= 17):
                return -3
        
        # Check preferred time
        if self.preferred_time == "morning" and 6 <= hour < 12:
            return 3
        elif self.preferred_time == "afternoon" and 12 <= hour < 17:
            return 3
        elif self.preferred_time == "evening" and 17 <= hour < 22:
            return 3
        elif self.preferred_time == "any":
            return 1
        
        return 0
    
    def get_urgency_bonus(self):
        """Calculate bonus based on deadline proximity"""
        if not self.deadline:
            return 0
        
        try:
            deadline_date = datetime.strptime(self.deadline, "%Y-%m-%d")
            days_until = (deadline_date - datetime.now()).days
            
            if days_until < 0:
                return 10  # Overdue!
            elif days_until == 0:
                return 8   # Due today
            elif days_until <= 2:
                return 5   # Due in 1-2 days
            elif days_until <= 7:
                return 2   # Due this week
            else:
                return 0   # Not urgent
        except:
            return 0
    
    def mark_attempted(self, successful=True):
        """Update task statistics after an attempt"""
        self.attempt_count += 1
        self.last_attempted = datetime.now()
        
        if successful:
            self.completed = True
            # Successful attempts increase success rate
            self.success_rate = min(1.0, self.success_rate + 0.1)
        else:
            # Failed attempts decrease success rate
            self.success_rate = max(0.1, self.success_rate - 0.2)
    
    def check_constraints(self, current_conditions):
        """Check if all constraints are satisfied with current conditions"""
        if "date" in self.constraints or "start_date" in self.constraints:
            target_date_str = self.constraints.get("date") or self.constraints.get("start_date")
            try:
                target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
                current_date = datetime.now().date()
                if current_date < target_date:
                    return False # Too early
            except:
                pass

        if "requires" in self.constraints:
            required_item = self.constraints["requires"]
            if isinstance(required_item, list):
                if not all(item in current_conditions.get("available_resources", []) for item in required_item):
                    return False
            elif required_item not in current_conditions.get("available_resources", []):
                return False
        
        if "max_time" in self.constraints:
            if self.duration > self.constraints["max_time"]:
                return False
        
        # Check time constraints
        current_time = datetime.now()
        if not self.is_time_suitable(current_time):
            return False
        
        return True
    
    def is_time_suitable(self, current_time):
        """Check if current time is within allowed time constraints"""
        hour = current_time.hour
        
        if "time_constraints" in self.constraints:
            time_constraints = self.constraints["time_constraints"]
            if "morning_only" in time_constraints and hour >= 12:
                return False
            if "evening_only" in time_constraints and hour < 17:
                return False
            if "office_hours" in time_constraints and (hour < 9 or hour >= 17):
                return False
            if "weekends_only" in time_constraints:
                weekday = current_time.weekday()  # 0=Monday, 6=Sunday
                if weekday < 5:  # Monday-Friday
                    return False
        
        return True
    
    def __repr__(self):
        return (f"<Task: {self.name[:20]:20} | "
                f"Type: {self.task_type:10} | "
                f"Score: {self.score:5.1f} | "
                f"Emotions: {self.emotion_fit} | "
                f"Duration: {self.duration}min>")