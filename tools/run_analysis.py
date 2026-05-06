import yaml
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run all generate_* analysis scripts.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the configuration YAML file.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.")
        return

    # Find all generate_*.py scripts in the current directory
    current_dir = Path(__file__).parent
    generate_scripts = sorted(list(current_dir.glob("generate_*.py")))

    if not generate_scripts:
        print("No 'generate_*.py' scripts found in the current directory.")
        return

    print(f"Found {len(generate_scripts)} plotting scripts. Executing them sequentially...")

    for script_path in generate_scripts:
        script_name = script_path.name
        print(f"\n--- Executing {script_name} ---")
        try:
            # Pass the config file to each script
            subprocess.run(['python', str(script_path), '--config', str(config_path)], check=True)
            print(f"--- Successfully executed {script_name} ---")
        except FileNotFoundError:
            print(f"Error: Python interpreter not found or script '{script_name}' does not exist.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script_name}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while executing {script_name}: {e}")

    print("\nAll specified plotting scripts have been attempted.")

if __name__ == '__main__':
    main()
