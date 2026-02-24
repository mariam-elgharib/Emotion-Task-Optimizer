def check_libraries():
    # Map display name → actual import name
    packages = {
        "fer": "fer",
        "opencv-python": "cv2",  # correct import name for OpenCV
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy"
    }

    print("\nChecking installed Python libraries...\n")
    
    for display_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"{display_name} ✅ Installed")
        except ImportError:
            print(f"{display_name} ❌ NOT Installed")
    
    print("\nDone!\n")

if __name__ == "__main__":
    check_libraries()
