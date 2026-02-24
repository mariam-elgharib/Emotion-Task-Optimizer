from task import Task

def collect_user_preferences():
    profile = {}
    
    print("\nAre you:")
    print("1. Student")
    print("2. Employee")
    user_type = input("Choice: ")

    tasks = []

    if user_type == "1":
        print("\nEnter Academic Tasks (type 'done' to stop)")
        while True:
            name = input("Task name: ")
            if name == "done": break
            priority = int(input("Priority (1-5): "))
            tasks.append(Task(name, "academic", priority, 
                              ["neutral", "focused", "happy"], "high"))

    if user_type == "2":
        print("\nEnter Work Tasks (type 'done' to stop)")
        while True:
            name = input("Task name: ")
            if name == "done": break
            priority = int(input("Priority (1-5): "))
            tasks.append(Task(name, "work", priority,
                              ["neutral", "focused"], "high"))

    print("\nEnter Personal / Life Activities")
    while True:
        name = input("Activity name: ")
        if name == "done": break
        tasks.append(Task(name, "personal", 3,
                          ["sad", "tired", "angry"], "low"))

    print("\nEnter Break Activities")
    while True:
        name = input("Break activity: ")
        if name == "done": break
        tasks.append(Task(name, "break", 2,
                          ["tired", "angry"], "low"))

    profile["tasks"] = tasks
    return profile
