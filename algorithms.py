# algorithms.py
import random
from datetime import datetime
import copy
from heapq import heappush, heappop

# ==============================================
# SMARTER CSP WITH MRV AND DEGREE HEURISTIC
# ==============================================

class CSP:
    """
    Constraint Satisfaction Problem Solver
    Uses MRV (Minimum Remaining Values) and Degree Heuristic
    """
    def __init__(self, tasks, emotion, current_conditions=None):
        self.tasks = tasks
        self.emotion = emotion
        self.conditions = current_conditions or {}
        self.variables = [task.name for task in tasks]
        self.domains = {task.name: [True, False] for task in tasks}  # True=recommend, False=don't recommend
        self.constraints = []
        self.task_map = {task.name: task for task in tasks}
        
        # Add constraints
        self.add_emotion_constraint()
        self.add_resource_constraint()
        self.add_time_constraint()
    
    def add_emotion_constraint(self):
        """Constraint: Task must match current emotion"""
        def constraint(assignment, variable, value):
            if value:  # If we're recommending this task
                task = self.task_map[variable]
                return self.emotion in task.emotion_fit
            return True  # It's okay to not recommend any task
        
        self.constraints.append(constraint)
    
    def add_time_constraint(self):
        """Constraint: Task must be time-appropriate"""
        def constraint(assignment, variable, value):
            if value:  # If we're recommending this task
                task = self.task_map[variable]
                current_time = self.conditions.get("current_time", datetime.now())
                return task.is_time_suitable(current_time)
            return True
        
        self.constraints.append(constraint)
    
    def add_resource_constraint(self):
        """Constraint: Must have required resources"""
        def constraint(assignment, variable, value):
            if value:  # If we're recommending this task
                task = self.task_map[variable]
                return task.check_constraints(self.conditions)
            return True
        
        self.constraints.append(constraint)
    
    def mrv_heuristic(self, assignment):
        """Minimum Remaining Values: Choose variable with fewest legal values"""
        unassigned = [var for var in self.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Calculate remaining values for each unassigned variable
        mrv_scores = []
        for variable in unassigned:
            legal_values = 0
            for value in self.domains[variable]:
                if self.is_consistent(assignment, variable, value):
                    legal_values += 1
            mrv_scores.append((legal_values, variable))
        
        # Return variable with minimum remaining values
        min_legal, best_var = min(mrv_scores, key=lambda x: x[0])
        return best_var
    
    def degree_heuristic(self, assignment):
        """Degree Heuristic: Choose variable with most constraints on remaining variables"""
        unassigned = [var for var in self.variables if var not in assignment]
        
        if not unassigned:
            return None
        
        # Calculate degree for each unassigned variable
        degree_scores = []
        for variable in unassigned:
            degree = 0
            task1 = self.task_map[variable]
            
            # Check constraints with other unassigned variables
            for other_var in unassigned:
                if variable != other_var:
                    task2 = self.task_map[other_var]
                    
                    # Check if tasks share constraints
                    # 1. Time constraints
                    time1_ok = task1.is_time_suitable(datetime.now())
                    time2_ok = task2.is_time_suitable(datetime.now())
                    if not time1_ok and not time2_ok:
                        degree += 1
                    
                    # 2. Resource constraints
                    if hasattr(task1, 'constraints') and hasattr(task2, 'constraints'):
                        req1 = task1.constraints.get("requires", "")
                        req2 = task2.constraints.get("requires", "")
                        if req1 and req2 and req1 == req2:
                            degree += 1
            
            degree_scores.append((degree, variable))
        
        # Return variable with maximum degree
        max_degree, best_var = max(degree_scores, key=lambda x: x[0])
        return best_var
    
    def is_consistent(self, assignment, variable, value):
        """Check if assignment is consistent with all constraints"""
        # Create temporary assignment
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value
        
        # Check all constraints
        for constraint in self.constraints:
            if not constraint(temp_assignment, variable, value):
                return False
        
        return True
    
    def backtrack(self, assignment):
        """Backtracking search with heuristics"""
        # If all variables assigned, return assignment
        if len(assignment) == len(self.variables):
            return assignment
        
        # Select variable using MRV, tie-break with Degree Heuristic
        var = self.mrv_heuristic(assignment)
        
        # If MRV didn't find a variable (shouldn't happen), use Degree Heuristic
        if var is None:
            var = self.degree_heuristic(assignment)
        
        # Try values in order (True first, then False)
        for value in [True, False]:
            if self.is_consistent(assignment, var, value):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                del assignment[var]
        
        return None  # No solution
    
    def solve(self):
        """Solve CSP and return recommended tasks and warnings"""
        assignment = {}
        solution = self.backtrack(assignment)
        
        if solution is None:
            return [], ["CSP could not find a solution"]
        
        # Extract recommended tasks
        recommended_tasks = []
        warnings = []
        
        for task_name, recommended in solution.items():
            if recommended:  # If task is recommended
                task = self.task_map[task_name]
                recommended_tasks.append(task)
                
                # Check for warnings
                current_time = self.conditions.get("current_time", datetime.now())
                if not task.is_time_suitable(current_time):
                    warnings.append(f"{task.name}: Task may not be suitable for current time")
                
                # Check resource constraints
                if not task.check_constraints(self.conditions):
                    required = task.constraints.get("requires", "")
                    if required:
                        warnings.append(f"{task.name}: Requires {required}")
        
        return recommended_tasks, warnings

# ==============================================
# NEW CSP FUNCTIONS FOR CS PROJECT
# ==============================================

def csp_task(tasks, current_time=None):
    """
    CSP function that constraints based on deadline.
    Recommends tasks based on nearest deadline with warnings for urgency.
    
    Returns: (recommended_task, warning_message, urgency_level)
    """
    if not tasks:
        return None, "No tasks with deadlines found", "none"
    
    # Filter tasks that have deadlines
    tasks_with_deadlines = []
    for task in tasks:
        if task.deadline:
            tasks_with_deadlines.append(task)
    
    if not tasks_with_deadlines:
        return None, "No tasks have deadlines set", "none"
    
    # Parse deadlines and calculate urgency
    from datetime import datetime
    if current_time is None:
        current_time = datetime.now()
    
    urgency_categories = {
        "overdue": [],
        "due_today": [],
        "due_tomorrow": [],
        "due_week": [],
        "due_later": []
    }
    
    for task in tasks_with_deadlines:
        try:
            deadline_date = datetime.strptime(task.deadline, "%Y-%m-%d")
            days_until = (deadline_date - current_time).days
            
            if days_until < 0:
                urgency_categories["overdue"].append((task, days_until))
            elif days_until == 0:
                urgency_categories["due_today"].append((task, days_until))
            elif days_until == 1:
                urgency_categories["due_tomorrow"].append((task, days_until))
            elif days_until <= 7:
                urgency_categories["due_week"].append((task, days_until))
            else:
                urgency_categories["due_later"].append((task, days_until))
        except:
            continue
    
    # Select the most urgent task
    selected_task = None
    warning = ""
    urgency_level = "none"
    
    # Check categories in order of urgency
    for category in ["overdue", "due_today", "due_tomorrow", "due_week", "due_later"]:
        if urgency_categories[category]:
            # Sort by days_until (ascending for overdue, ascending for others)
            if category == "overdue":
                urgency_categories[category].sort(key=lambda x: abs(x[1]))  # Most overdue first
            else:
                urgency_categories[category].sort(key=lambda x: x[1])  # Earliest first
            
            selected_task = urgency_categories[category][0][0]
            urgency_level = category
            
            # Generate warning message
            days = urgency_categories[category][0][1]
            if category == "overdue":
                warning = f"This task is {abs(days)} days OVERDUE! You should complete it immediately."
            elif category == "due_today":
                warning = f"This task is DUE TODAY! Complete it now."
            elif category == "due_tomorrow":
                warning = f"This task is due TOMORROW. Consider starting it today."
            elif category == "due_week":
                warning = f"This task is due in {days} days. Plan accordingly."
            else:
                warning = f"This task is due in {days} days."
            break
    
    return selected_task, warning, urgency_level


def csp_preferences(tasks, current_conditions, use_mrv=True, use_dh=True):
    """
    CSP function for user preferences with time, resources, and emotion constraints.
    Uses MRV (Minimum Remaining Values) and Degree Heuristic.
    
    Args:
        tasks: List of tasks with user preferences
        current_conditions: Dict with current_time, available_resources, current_emotion
        use_mrv: Whether to use MRV heuristic
        use_dh: Whether to use Degree Heuristic
    
    Returns: List of recommended tasks sorted by fitness
    """
    if not tasks:
        return []
    
    # Extract current conditions
    current_time = current_conditions.get("current_time", datetime.now())
    available_resources = current_conditions.get("available_resources", [])
    current_emotion = current_conditions.get("current_emotion", "neutral")
    
    # Step 1: Filter tasks that pass all hard constraints
    valid_tasks = []
    excluded_reasons = []
    
    for task in tasks:
        reasons = []
        # Check resource constraint
        resource_valid = True
        if "requires" in task.constraints:
            required = task.constraints["requires"]
            if isinstance(required, list):
                for req in required:
                    if req not in available_resources:
                        resource_valid = False
                        break
            elif required not in available_resources:
                resource_valid = False

        # Check time constraint
        time_valid = True
        if "allowed_time" in task.constraints:
            allowed_time = task.constraints["allowed_time"]
            if isinstance(allowed_time, dict):
                start_hour = allowed_time.get("start", 0)
                end_hour = allowed_time.get("end", 24)
                current_hour = current_time.hour
                if not (start_hour <= current_hour < end_hour):
                    time_valid = False
        
        # Check preferred_time (used by default tasks)
        if "preferred_time" in task.constraints:
            pref = task.constraints["preferred_time"]
            current_hour = current_time.hour
            is_valid = True
            
            if pref == "morning":
                is_valid = (6 <= current_hour < 12)
            elif pref == "afternoon":
                is_valid = (12 <= current_hour < 17)
            elif pref == "evening":
                is_valid = (17 <= current_hour < 22)
            elif pref == "daylight":
                is_valid = (7 <= current_hour < 18)
            elif pref == "night":
                is_valid = (19 <= current_hour or current_hour < 5)
            # "any" is always valid
            
            if not is_valid:
                time_valid = False
        
        # Check emotion constraint
        # Manual preferences still need to match emotions, but get priority bonus in scoring
        emotion_valid = current_emotion in task.emotion_fit
        
        if resource_valid and time_valid and emotion_valid:
            valid_tasks.append(task)
        else:
            if not resource_valid: reasons.append("Resource constraint")
            if not time_valid: reasons.append("Time constraint")
            if not emotion_valid: reasons.append(f"Emotion mismatch ({current_emotion})")
            if reasons: excluded_reasons.append(f"**{task.name}**: {', '.join(reasons)}")
    
    # Step 2: Calculate remaining values for MRV heuristic
    if use_mrv:
        for task in valid_tasks:
            # Calculate how many constraints this task violates for other conditions
            # This is a simplified MRV - we count "soft" violations
            
            # Time flexibility: how strict is the time constraint?
            time_flexibility = 0
            if "allowed_time" in task.constraints:
                allowed_time = task.constraints["allowed_time"]
                if isinstance(allowed_time, dict):
                    start = allowed_time.get("start", 0)
                    end = allowed_time.get("end", 24)
                    time_flexibility = end - start  # Larger window = more flexible
            
            # Resource requirements: fewer requirements = more flexible
            resource_flexibility = 0
            if "requires" in task.constraints:
                required = task.constraints["requires"]
                if isinstance(required, list):
                    resource_flexibility = 10 - len(required)  # Fewer requirements = higher flexibility
                else:
                    resource_flexibility = 9  # Single requirement
            
            # Emotion fit: more emotions = more flexible
            emotion_flexibility = len(task.emotion_fit)
            
            # Combine flexibilities
            total_flexibility = time_flexibility + resource_flexibility + emotion_flexibility
            task.mrv_score = total_flexibility  # Higher = more flexible = more remaining values
    
    # Step 3: Calculate degree for Degree Heuristic
    if use_dh:
        for task in valid_tasks:
            # Count how many other tasks share constraints with this one
            degree = 0
            
            for other_task in valid_tasks:
                if task == other_task:
                    continue
                
                # Share time constraints?
                if ("allowed_time" in task.constraints and 
                    "allowed_time" in other_task.constraints):
                    task_time = task.constraints["allowed_time"]
                    other_time = other_task.constraints["allowed_time"]
                    
                    if isinstance(task_time, dict) and isinstance(other_time, dict):
                        task_start = task_time.get("start", 0)
                        task_end = task_time.get("end", 24)
                        other_start = other_time.get("start", 0)
                        other_end = other_time.get("end", 24)
                        
                        # Check for overlap
                        if not (task_end <= other_start or task_start >= other_end):
                            degree += 1
                
                # Share resource requirements?
                if ("requires" in task.constraints and 
                    "requires" in other_task.constraints):
                    task_req = task.constraints["requires"]
                    other_req = other_task.constraints["requires"]
                    
                    if isinstance(task_req, list) and isinstance(other_req, list):
                        common = set(task_req) & set(other_req)
                        if common:
                            degree += len(common)
                    elif task_req == other_req:
                        degree += 1
            
            task.degree_score = degree
    
    # Step 4: Sort tasks based on heuristics
    if use_mrv and use_dh:
        # Combine MRV and DH: prioritize tasks with low MRV (few options) and high degree (constraining others)
        for task in valid_tasks:
            # Normalize scores (inverse for MRV since we want low MRV)
            mrv_norm = 10 - getattr(task, 'mrv_score', 5)  # Default 5 if not calculated
            dh_norm = getattr(task, 'degree_score', 0)
            
            # Weighted combination (adjust weights as needed)
            task.heuristic_score = 0.6 * mrv_norm + 0.4 * dh_norm
            
        valid_tasks.sort(key=lambda x: x.heuristic_score, reverse=True)
    
    elif use_mrv:
        valid_tasks.sort(key=lambda x: getattr(x, 'mrv_score', 0))
    
    elif use_dh:
        valid_tasks.sort(key=lambda x: getattr(x, 'degree_score', 0), reverse=True)
    
    # Step 5: Apply soft constraints and score
    scored_tasks = []
    for task in valid_tasks:
        # Calculate fitness score
        fitness = 0
        
        # Time fitness: how close to optimal time?
        if "allowed_time" in task.constraints and isinstance(task.constraints["allowed_time"], dict):
            allowed_time = task.constraints["allowed_time"]
            optimal_hour = allowed_time.get("optimal", (allowed_time.get("start", 0) + allowed_time.get("end", 24)) / 2)
            time_diff = abs(current_time.hour - optimal_hour)
            time_fitness = max(0, 10 - time_diff)  # Higher if closer to optimal
            fitness += time_fitness * 0.4
        
        # Resource fitness: prefer tasks using available resources
        if "requires" in task.constraints:
            required = task.constraints["requires"]
            if isinstance(required, list):
                available_count = sum(1 for req in required if req in available_resources)
                resource_fitness = (available_count / len(required)) * 10
            else:
                resource_fitness = 10 if required in available_resources else 0
            fitness += resource_fitness * 0.3
        
        # Emotion fitness: exact match vs close emotions
        emotion_fitness = 10 if current_emotion in task.emotion_fit else 5
        fitness += emotion_fitness * 0.3
        
        # User Preference Bonus: Give HUGE priority to custom user preferences
        if getattr(task, 'is_preference', False):
             fitness += 30
        
        task.fitness_score = fitness
        scored_tasks.append((fitness, task))
    
    # Sort by fitness score
    scored_tasks.sort(key=lambda x: x[0], reverse=True)
    
    # Return sorted list of tasks AND excluded reasons
    return [task for _, task in scored_tasks], excluded_reasons


def create_preference_task(name, emotions, time_range, required_resources, base_priority=5):
    """
    Helper function to create a preference task with proper constraints.
    
    Args:
        name: Task name
        emotions: List of suitable emotions
        time_range: Tuple or dict of allowed time (e.g., (9, 17) or {"start": 9, "end": 17})
        required_resources: List or string of required resources
        base_priority: Base priority (1-10)
    """
    from task import Task
    
    constraints = {}
    
    # Parse time range
    if isinstance(time_range, tuple) and len(time_range) == 2:
        constraints["allowed_time"] = {"start": time_range[0], "end": time_range[1]}
    elif isinstance(time_range, dict):
        constraints["allowed_time"] = time_range
    
    # Parse required resources
    if required_resources:
        if isinstance(required_resources, list):
            constraints["requires"] = required_resources
        else:
            constraints["requires"] = [required_resources]
    
    # Create task
    task = Task(
        name=name,
        base_priority=base_priority,
        category="user_preference",
        duration=30,  # Default duration
        emotion_fit=emotions if isinstance(emotions, list) else [emotions],
        task_type="preference",
        constraints=constraints
    )
    
    return task

# ==============================================
# MAIN FILTERING AND SELECTION ALGORITHMS
# ==============================================

def apply_strict_constraints(tasks, conditions, current_emotion):
    """
    Standardizes constraint filtering across all algorithms.
    Filters tasks by emotion, time, and resources.
    """
    if not tasks:
        return []
        
    current_time = conditions.get("current_time", datetime.now())
    available_resources = conditions.get("available_resources", [])
    
    valid_tasks = []
    for task in tasks:
        # 1. Emotion constraint
        if current_emotion not in task.emotion_fit:
            # Check if it's a high priority must-do task (allow slight mismatch for these)
            if not (hasattr(task, 'task_type') and task.task_type == "must_do" and task.base_priority >= 7):
                continue
        
        # 2. Time constraint
        if not task.is_time_suitable(current_time):
            continue
            
        # 3. Resource constraint
        if not task.check_constraints(conditions):
            continue
            
        valid_tasks.append(task)
        
    return valid_tasks

def csp_filter(tasks, emotion, current_conditions=None):
    """
    CSP Filter using constraint satisfaction problem solving
    Returns: (recommended_tasks, warnings)
    """
    if not tasks:
        return [], []
    
    # Create CSP instance and solve
    csp = CSP(tasks, emotion, current_conditions)
    recommended_tasks, warnings = csp.solve()
    
    return recommended_tasks, warnings

def greedy(tasks, emotion=None, current_conditions=None):
    """Production-ready greedy algorithm using compute_score() as heuristic"""
    if not tasks:
        return None
    
    # Extract current conditions
    current_time = datetime.now()
    current_energy = 5
    preference_bonus = 0
    urgency_bonus = 0
    
    if current_conditions:
        current_time = current_conditions.get("current_time", datetime.now())
        current_energy = current_conditions.get("current_energy", 5)
        preference_bonus = current_conditions.get("preference_bonus", 0)
        urgency_bonus = current_conditions.get("urgency_bonus", 0)
    
    # Apply strict constraints
    valid_tasks = apply_strict_constraints(tasks, current_conditions or {}, emotion or "neutral")
    
    if not valid_tasks:
        return None
        
    # Compute scores for valid tasks
    for task in valid_tasks:
        task.heuristic_score = task.compute_score(
            current_emotion=emotion if emotion else "neutral",
            current_time=current_time,
            current_energy=current_energy,
            preference_bonus=preference_bonus,
            urgency_bonus=urgency_bonus
        )
    
    # Find best task with HIGHEST heuristic score (Greedy: choose best estimated)
    best_task = None
    best_score = float('-inf')
    
    for task in valid_tasks:
        current_score = getattr(task, 'heuristic_score', 0)
        
        if current_score > best_score:
            best_score = current_score
            best_task = task
        elif current_score == best_score and best_task:
            # Tie-breaking: use priority, then duration
            if task.base_priority > best_task.base_priority:
                best_task = task
            elif task.base_priority == best_task.base_priority:
                if task.duration < best_task.duration:
                    best_task = task
    
    return best_task

def hill_climbing(tasks, current_conditions=None, max_iterations=5):
    """
    Hill climbing algorithm for preference/mood tasks.
    Finds the best task by exploring similar tasks in the neighborhood.
    """
    if not tasks:
        return None
    
    # Extract current conditions
    current_emotion = "neutral"
    current_energy = 5
    
    if current_conditions:
        current_emotion = current_conditions.get("current_emotion", "neutral")
        current_energy = current_conditions.get("current_energy", 5)
    
    # Apply strict constraints
    valid_tasks = apply_strict_constraints(tasks, current_conditions or {}, current_emotion)
    
    if not valid_tasks:
        return None
    
    # Ensure all tasks have scores computed
    for task in valid_tasks:
        task.compute_score(current_emotion, current_energy=current_energy)
    
    # Start with the task that has the highest score
    current_best = max(valid_tasks, key=lambda t: t.score)
    
    for iteration in range(max_iterations):
        # Find neighboring tasks (similar tasks)
        neighbors = []
        
        for task in tasks:
            if task == current_best:
                continue
            
            # Calculate similarity between tasks
            similarity = calculate_task_similarity(current_best, task)
            
            # Consider as neighbor if similar enough (≥ 60%)
            if similarity >= 0.6:
                neighbors.append(task)
        
        if not neighbors:
            break  # No more neighbors to explore
        
        # Find best neighbor
        best_neighbor = max(neighbors, key=lambda t: t.score)
        
        # Move to neighbor only if it's better (hill climbing principle)
        if best_neighbor.score > current_best.score:
            current_best = best_neighbor
        else:
            break  # No improvement found, reached local optimum
    
    return current_best

def calculate_task_similarity(task1, task2):
    """Calculate similarity between two preference/mood tasks"""
    similarity = 0.0
    factors = 0
    
    # 1. Emotion fit similarity - weighted higher
    if hasattr(task1, 'emotion_fit') and hasattr(task2, 'emotion_fit'):
        emotion_overlap = len(set(task1.emotion_fit) & set(task2.emotion_fit))
        emotion_total = len(set(task1.emotion_fit) | set(task2.emotion_fit))
        if emotion_total > 0:
            similarity += (emotion_overlap / emotion_total) * 2
            factors += 2
    
    # 2. Duration similarity
    if hasattr(task1, 'duration') and hasattr(task2, 'duration'):
        if task1.duration > 0 and task2.duration > 0:
            duration_ratio = min(task1.duration, task2.duration) / max(task1.duration, task2.duration)
            similarity += duration_ratio
            factors += 1
    
    # 3. Priority similarity
    if hasattr(task1, 'base_priority') and hasattr(task2, 'base_priority'):
        priority_diff = abs(task1.base_priority - task2.base_priority)
        similarity += (10 - priority_diff) / 10
        factors += 1
    
    # 4. Task type similarity
    if hasattr(task1, 'task_type') and hasattr(task2, 'task_type'):
        if task1.task_type == task2.task_type:
            similarity += 1.0
        factors += 1
    
    # 5. Category similarity
    if hasattr(task1, 'category') and hasattr(task2, 'category'):
        if task1.category == task2.category:
            similarity += 1.0
        factors += 1
    
    # Return weighted average similarity
    return similarity / factors if factors > 0 else 0.5

def stochastic(tasks, emotion=None, current_conditions=None):
    """
    PURE RANDOM selection from valid tasks that pass strict constraints.
    """
    if not tasks:
        return None
    
    # Apply strict constraints
    valid_tasks = apply_strict_constraints(tasks, current_conditions or {}, emotion or "neutral")
    
    if not valid_tasks:
        return None
    
    # If only one task, return it
    if len(valid_tasks) == 1:
        return valid_tasks[0]
    
    # PURE RANDOM selection
    return random.choice(valid_tasks)


def mini_a_star(tasks, current_emotion, current_conditions=None, max_sequence_length=3, user_preferences=None):
    """
    A* inspired search for a sequence of 3 tasks.
    """
    # 1. Gather all tasks
    pool = tasks.copy() if tasks else []
    if user_preferences:
        pool.extend(user_preferences)
        
    if not pool:
        return []
        
    # 2. Apply strict constraints
    valid_pool = apply_strict_constraints(pool, current_conditions or {}, current_emotion)
    
    if not valid_pool:
        return []
        
    # 3. Remove duplicates
    unique_tasks = {}
    for task in valid_pool:
        if task.name not in unique_tasks:
            unique_tasks[task.name] = task
    all_tasks = list(unique_tasks.values())
    
    fatigue_score = 0
    if current_conditions:
        energy = current_conditions.get("current_energy", 5)
        fatigue_score = 10 - energy
    
    # g(n): Cost of current task - KEEPING THE SAME AS BEFORE
    def g_score(task, emotion, is_first=False):
        cost = 10 - task.base_priority                     # Higher priority → lower cost
        if emotion not in task.emotion_fit:
            cost += 5                                  # Emotion mismatch penalty
        cost += fatigue_score * 0.5                    # Fatigue penalty
        if is_first:
            cost -= 2                                  # Encourage good starting task
        
        # Special handling for preference tasks
        if hasattr(task, 'task_type') and task.task_type == "preference":
            # Lower cost for preference tasks that match current emotion well
            if emotion in task.emotion_fit:
                cost -= 3
            # Consider time constraints for preference tasks
            if "allowed_time" in task.constraints:
                current_hour = datetime.now().hour
                time_constraint = task.constraints["allowed_time"]
                if isinstance(time_constraint, dict):
                    start = time_constraint.get("start", 0)
                    end = time_constraint.get("end", 24)
                    if not (start <= current_hour < end):
                        cost += 5  # Penalty if not in allowed time
        
        return max(0, cost)  # Ensure non-negative

    # h(n): Estimated future benefit - USING compute_score() as heuristic
    def h_score(task, predicted_emotion):
        # Use compute_score() as the heuristic estimate
        # We want to convert score (higher = better) to cost (lower = better) for A*
        heuristic_value = task.compute_score(
            current_emotion=predicted_emotion,  # Use predicted emotion for h(n)
            current_time=datetime.now(),
            current_energy=10 - fatigue_score,  # Convert fatigue to energy (0-10 scale)
            preference_bonus=0,
            urgency_bonus=0
        )
        
        # In A*, we want to MINIMIZE f = g + h
        # But compute_score() returns HIGHER = better
        # So we need to invert: better task = lower h value
        
        # Method 1: Use negative of score (simple)
        # return -heuristic_value
        
        # Method 2: Normalize to a reasonable range (0-100) and invert
        # compute_score() typically returns ~5-50 range
        # So: h = 50 - score (making better tasks have lower h)
        return 50 - heuristic_value

    # Predict emotion after task - KEEPING SAME
    def predict_next_emotion(task, emotion):
        if task.category in ["break", "personal", "life", "mood_enhancer", "user_preference"]:
            # Mood-improving tasks
            if emotion in ["sad", "angry", "tired", "stressed"]:
                return "neutral"
            elif emotion in ["neutral", "happy"]:
                return "happy"
            else:
                return emotion
        
        if task.category in ["academic", "work", "todo_must"]:
            # Work/study tasks
            if emotion in ["neutral", "focused"]:
                return "focused"
            elif emotion in ["sad", "angry", "tired"]:
                return "tired"
            else:
                return emotion
        
        # Default: return same emotion or neutral
        return "neutral" if emotion in ["sad", "angry", "tired"] else emotion

    # Search Node - KEEPING SAME
    class Node:
        def __init__(self, task, emotion, g, h, parent=None):
            self.task = task
            self.emotion_after = predict_next_emotion(task, emotion)
            self.g = g
            self.h = h
            self.f = g + h  # Total cost to minimize
            self.parent = parent
            self.depth = 1 if parent is None else parent.depth + 1

        def __lt__(self, other):
            return self.f < other.f
        
        def __eq__(self, other):
            if not isinstance(other, Node):
                return False
            return (self.task == other.task and 
                    self.g == other.g and 
                    self.h == other.h)

        def sequence(self):
            seq = []
            node = self
            while node:
                seq.insert(0, node.task)
                node = node.parent
            return seq

    # Initialize Open Set
    open_set = []

    for task in all_tasks:
        g = g_score(task, current_emotion, is_first=True)
        h = h_score(task, predict_next_emotion(task, current_emotion))
        node = Node(task, current_emotion, g, h)
        heappush(open_set, (g + h, id(node), node))

    best_sequence = None
    best_score = float("inf")

    # Mini A* Search Loop
    while open_set:
        _, _, node = heappop(open_set)

        if node.depth >= max_sequence_length:
            if node.f < best_score:
                best_score = node.f
                best_sequence = node.sequence()
            continue

        for next_task in all_tasks:
            if next_task in node.sequence():
                continue  # Avoid repetition
            
            # Check if emotion matches well for the predicted state
            predicted_emotion = node.emotion_after
            if predicted_emotion not in next_task.emotion_fit:
                # Still allow high priority tasks
                if not (hasattr(next_task, 'task_type') and next_task.task_type == "must_do" and next_task.base_priority >= 7):
                    continue

            g_new = node.g + g_score(next_task, node.emotion_after)
            h_new = h_score(next_task, predict_next_emotion(next_task, node.emotion_after))

            # Create node
            new_node = Node(next_task, node.emotion_after, g_new, h_new, node)
            heappush(open_set, (g_new + h_new, id(new_node), new_node))

    # Return best sequence or fallback
    if best_sequence and len(best_sequence) >= 2:
        return best_sequence[:max_sequence_length]
    
    # Fallback: use greedy as fallback (consistent with heuristic)
    fallback_task = greedy(all_tasks, current_emotion)
    return [fallback_task] if fallback_task else []

def fallback_sequence_with_preferences(all_tasks, emotion, user_preferences=None, length=3):
    """
    Simple emotion-aware fallback sequence that includes user preferences
    """
    sequence = []
    
    # Filter tasks that match current emotion
    matching_tasks = [t for t in all_tasks if emotion in t.emotion_fit]
    
    if not matching_tasks:
        # If no tasks match emotion, use all tasks
        matching_tasks = all_tasks
    
    if emotion in ["sad", "angry", "tired", "stressed"]:
        # Mood improvement → productivity
        mood_tasks = [t for t in matching_tasks if t.category in ["break", "personal", "life", "mood_enhancer", "user_preference"]]
        work_tasks = [t for t in matching_tasks if t.category in ["academic", "work", "todo_must"]]

        if mood_tasks:
            # Sort by priority and duration (shorter first for mood tasks)
            mood_tasks.sort(key=lambda x: (x.base_priority, -x.duration), reverse=True)
            sequence.extend(mood_tasks[:1])
        
        if work_tasks and len(sequence) < length:
            work_tasks.sort(key=lambda x: x.base_priority, reverse=True)
            sequence.extend(work_tasks[:length - len(sequence)])

    else:
        # Neutral/good emotions: Focus on productivity first, then preferences
        productive_tasks = [t for t in matching_tasks if t.category in ["academic", "work", "todo_must"]]
        preference_tasks = [t for t in matching_tasks if t.category in ["user_preference", "mood_enhancer"]]
        
        if productive_tasks:
            productive_tasks.sort(key=lambda x: x.base_priority, reverse=True)
            sequence.extend(productive_tasks[:min(2, length)])
        
        if preference_tasks and len(sequence) < length:
            preference_tasks.sort(key=lambda x: x.base_priority, reverse=True)
            sequence.extend(preference_tasks[:length - len(sequence)])

    # Fill remaining slots if needed
    if len(sequence) < length:
        remaining = [t for t in matching_tasks if t not in sequence]
        remaining.sort(key=lambda x: x.base_priority, reverse=True)
        sequence.extend(remaining[:length - len(sequence)])

    return sequence[:length]

def multi_objective_optimization(tasks, weights=None):
    """
    Optimize across multiple objectives:
    - Score (quality)
    - Duration (time)
    - Success rate (reliability)
    """
    if not tasks:
        return []
    
    if weights is None:
        weights = {'score': 0.5, 'duration': 0.3, 'success': 0.2}
    
    optimized = []
    for task in tasks:
        # Normalize values to 0-1 range
        max_score = max(t.score for t in tasks) if tasks else 1
        score_norm = task.score / max_score if max_score > 0 else 0
        
        # Duration: shorter is better, inverse relationship
        max_duration = max(t.duration for t in tasks) if tasks else 1
        duration_norm = 1 - (task.duration / max_duration if max_duration > 0 else 0)
        
        # Success rate already 0-1
        success_norm = task.success_rate
        
        # Combined score
        combined = (weights['score'] * score_norm +
                   weights['duration'] * duration_norm +
                   weights['success'] * success_norm)
        
        optimized.append((combined, task))
    
    optimized.sort(key=lambda x: x[0], reverse=True)
    return [task for score, task in optimized[:3]]

def analyze_csp_failure(tasks, emotion, conditions):
    """Analyze why CSP filtered out tasks"""
    constraints_pass = []
    
    # Check each constraint
    emotion_matches = [t for t in tasks if emotion in t.emotion_fit]
    
    time_suitable = [t for t in emotion_matches if t.is_time_suitable(conditions.get("current_time", datetime.now()))]
    
    constraints_pass = [t for t in time_suitable if t.check_constraints(conditions)]
    
    return constraints_pass