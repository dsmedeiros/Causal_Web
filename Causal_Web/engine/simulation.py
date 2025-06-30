# engine/simulation.py

import time
from threading import Thread
from config import Config


def simulation_loop():
    def run():
        while True:
            if Config.is_running:
                Config.current_tick += 1
                print(f"Tick: {Config.current_tick}")

                if Config.max_ticks and Config.current_tick >= Config.max_ticks:
                    Config.is_running = False

                time.sleep(Config.tick_rate)
            else:
                time.sleep(0.1)  # Wait until resumed

    Thread(target=run, daemon=True).start()

