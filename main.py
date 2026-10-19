
import agent


def main():
    print("Clients :", ", ".join(agent.USERS))
    user = ""
    while user not in agent.USERS:
        user = input("Connecté en tant que : ").strip().capitalize()

    print("Tape 'quit' pour quitter.\n")
    while True:
        question = input("Vous : ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break
        try:
            result = agent.ask(user, question)
            for call in result["tool_calls"]:
                print(f"  [outil] {call['name']}({call['args']})")
            print("\nAssistant :", result["answer"], "\n")
        except Exception as e:
            print("\nErreur :", e, "\n")


if __name__ == "__main__":
    main()
