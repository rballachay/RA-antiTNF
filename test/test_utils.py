from RAantiTNF.utils import NonResponseRA


def test(testpath="data/DREAM_RA_Responders_PhenoCov_Full.txt"):
    """test that the eular criteria here matches
    what we have in the data
    """
    import pandas as pd

    test = pd.read_csv(testpath, sep=" ")

    non_response_test = tuple(
        map(NonResponseRA(), test["baselineDAS"], test["Response.deltaDAS"])
    )
    non_response_real = tuple(test["Response.NonResp"].tolist())
    assert non_response_test == non_response_real
    """
    for index, (first, second) in enumerate(zip(non_response_test, non_response_real)):
        if first != second:
            print(index, first, second)
            print(
                test.loc[index, "baselineDAS"] + test.loc[index, "Response.deltaDAS"],
                test.loc[index, "Response.deltaDAS"],
            )
    """


if __name__ == "__main__":
    test()
