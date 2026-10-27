from f1tyre import serving


def main():
    serving.refresh()
    print(f"saved {serving.ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
