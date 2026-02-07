from usc.mem.template_miner_drain3 import mine_chunk_lines

def run():
    chunks = [
        "User Bob logged in from 10.0.0.1\nLatency=0.032s",
        "User Alice logged in from 10.0.0.2\nLatency=0.028s",
    ]
    templates, params = mine_chunk_lines(chunks)

    print("TEMPLATES:")
    for t in templates:
        print("---")
        print(t)

    print("\nPARAMS:")
    for p in params:
        print(p)

if __name__ == "__main__":
    run()
