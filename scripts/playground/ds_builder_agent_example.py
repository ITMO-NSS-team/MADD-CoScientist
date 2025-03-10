from ChemCoScientist.agents.agents import dataset_builder_agent


def main():
    querys = [
        """
        Find molecules that contain all following properties:
        ChEMBL ID
        Name
        Molecular Weight (between 250 and 500 Da)
        AlogP (between -2 and 5)
        Polar Surface Area (PSA) (between 20 and 150 Å²)
        #RO5 Violations (exactly 0 or 1)
        CX LogP (between -1 and 6)
        Aromatic Rings (between 0 and 6)
        Heavy Atoms (between 15 and 20)
        Molecular Formula
        """,
        "Connections with a number of rotatable bonds of no more than 5 and a positive LogD are required.",
        "Find molecules with positive LogD.",
        "Molecules with a polar surface area (PSA) of less than 100 are needed",
    ]

    for q in querys:
        print("========")
        print("Query:", q)
        res = dataset_builder_agent({"pending_tasks": [q], "responses": []})
        print(res["responses"][0])
        print(res["responses"][0].iloc[0])


if __name__ == "__main__":
    main()
