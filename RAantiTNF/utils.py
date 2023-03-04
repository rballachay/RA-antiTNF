class NonResponseRA:
    """see the full explanation of the EULAR critera at
    https://core.ac.uk/reader/16112664?utm_source=linkout
    """

    # we have defined zero as response and none
    # as non response
    NonReponses = {"good": 0, "moderate": 0, "none": 1}

    def __call__(self, baseline_das, delta_das):
        # we have eular categories, but then we also need to
        # determine whether or not we categorize a response
        end_das = baseline_das - delta_das
        eular = self.__eular__cat(end_das, delta_das)

        # convert category to binary response
        return self.NonReponses[eular]

    @staticmethod
    def __eular__cat(end_das, delta_das):
        """used elif instead of else for clarity,"""
        # if baseline das is already low
        if end_das <= 3.2:
            if delta_das > 1.2:
                return "good"
            elif delta_das > 0.6:
                return "moderate"
            elif delta_das <= 0.6:
                return "none"

        # if baseline das is moderate
        elif end_das <= 5.1:
            if delta_das >= 0.6:
                return "moderate"
            elif delta_das < 0.6:
                return "none"

        # if baseline das is high
        # at this point, we could just use
        # else, this is for readability
        elif end_das > 5.1:
            if delta_das >= 1.2:
                return "moderate"
            elif delta_das < 1.2:
                return "none"


def read_snps(path, snp_dict) -> dict:
    """read in the text files from guanlab that describe which snps belong"""
    out_dict = {}
    for drug, file in snp_dict.items():
        snppath = path / file
        with open(snppath, "r") as txtfile:
            lines = txtfile.read()
            snps = list(filter(lambda x: x, lines.split("\t")))
            snps = [i for x in snps for i in x.split("\n")]
            snps = list(filter(lambda x: x, snps))

        out_dict[drug] = snps
    return out_dict
