# default_tasks.py
from task import Task

def get_default_mood_tasks():
    """Create default mood-changing tasks if user doesn't provide any"""
    default_tasks = [
        Task(
            name="Listen to uplifting music",
            base_priority=6,
            category="mood_enhancer",
            duration=15,
            emotion_fit=["sad", "angry", "fear"],
            task_type="mood_changer",
            constraints={
                "requires": "headphones",
                "max_time": 30,
                "energy_required": 3,
                "preferred_time": "any"
            }
        ),
        Task(
            name="Take a short walk",
            base_priority=7,
            category="mood_enhancer",
            duration=20,
            emotion_fit=["sad", "angry", "fear"],
            task_type="mood_changer",
            constraints={
                "requires": "outdoors",
                "max_time": 30,
                "energy_required": 5,
                "preferred_time": "daylight"
            }
        ),
        Task(
            name="Watch funny videos",
            base_priority=5,
            category="mood_enhancer",
            duration=10,
            emotion_fit=["sad", "angry"],
            task_type="mood_changer",
            constraints={
                "requires": "internet",
                "max_time": 20,
                "energy_required": 2,
                "preferred_time": "any"
            }
        ),
        Task(
            name="Deep breathing exercise",
            base_priority=8,
            category="mood_enhancer",
            duration=5,
            emotion_fit=["angry", "fear", "sad"],
            task_type="mood_changer",
            constraints={
                "max_time": 10,
                "energy_required": 1,
                "preferred_time": "any"
            }
        ),
        Task(
            name="Drink water and stretch",
            base_priority=6,
            category="mood_enhancer",
            duration=5,
            emotion_fit=["sad", "neutral"],
            task_type="mood_changer",
            constraints={
                "requires": "water",
                "max_time": 10,
                "energy_required": 2,
                "preferred_time": "any"
            }
        )
    ]
    
    return default_tasks