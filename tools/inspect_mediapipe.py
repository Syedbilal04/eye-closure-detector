import mediapipe as mp


def main():
    print("file", getattr(mp, "__file__", None))
    print("version", getattr(mp, "__version__", None))
    print("has_solutions", hasattr(mp, "solutions"))
    print("has_tasks", hasattr(mp, "tasks"))
    print("dir_contains_solutions", "solutions" in dir(mp))
    print("dir_head", list(dir(mp))[:50])

    try:
        s = mp.solutions
        print("solutions", s)
    except Exception as e:
        print("solutions_error", type(e).__name__, str(e))

    try:
        from mediapipe.tasks.python import vision

        print("import mediapipe.tasks.python.vision OK", vision)
    except Exception as e:
        print("tasks.vision_error", type(e).__name__, str(e))


if __name__ == "__main__":
    main()
