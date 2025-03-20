from ChemCoScientist.agents.agents import ml_dl_agent


def main():
    querys = [
        "Predict Polar Surface Area by model 'test_case' for molecule: 'Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1'",
        "Predict Polar Surface Area for molecule Fc1cc(F)c2ccc(Oc3cncc4nnc(-c5ccc(OC(F)F)cc5)n34)cc2c1.",
        "Run train on dataset with next path: /Users/alina/Desktop/ИТМО/ChemCoScientist/ChemCoScientist/dataset_handler/chembl/test.csv, name for case 'run_by_llm_2.0', feature is Smiles, target is Polar Surface Area",
        "Is ready model 'run_by_llm_2.0'? Or not?",
    ]

    for q in querys:
        print("========")
        print("Query:", q)
        res = ml_dl_agent({"pending_tasks": [q], "responses": []})


if __name__ == "__main__":
    main()
