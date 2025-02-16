if __name__ == "__main__":
    import argparse
    from wu_2025.main import main

    parser = argparse.ArgumentParser(
        description="Run a seizure detection algorithm on an EDF file and store computed seizure annotations to a TSV."
    )
    parser.add_argument("input", help="Path to the input EDF EEG file.")
    parser.add_argument("output", help="Path to the output TSV annotation file.")

    args = parser.parse_args()
    main(args.input, args.output)