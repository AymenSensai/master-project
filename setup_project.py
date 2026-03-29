import os

def setup():
    # Directories needed in the project root
    dirs = [
        'checkpoints',
        'data/TuftsFaceDatabase/TD_RGB',
        'data/TuftsFaceDatabase/TD_IR',
        'gallery'
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
        else:
            print(f"Directory already exists: {d}")

    # Create dummy .gitkeep files to keep folders in some version control systems
    for d in dirs:
        keep_file = os.path.join(d, '.gitkeep')
        if not os.path.exists(keep_file):
            with open(keep_file, 'w') as f:
                pass
            print(f"Created .gitkeep in {d}")

if __name__ == "__main__":
    setup()
    print("\nProject structure initialized successfully.")
