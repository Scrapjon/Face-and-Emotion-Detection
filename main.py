from app import App

def main():
    app = App(db_path="debug_data")
    app.run()

if __name__ == "__main__":
    main()