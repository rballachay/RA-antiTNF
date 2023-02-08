class NonResponseRA:
    """see the full explanation of the EULAR critera at
    https://core.ac.uk/reader/16112664?utm_source=linkout
    """

    # we have defined zero as response and none
    # as non response
    NonReponses = {"good": 0, "moderate": 0, "none": 1}

    @classmethod
    def __call__(cls, baseline_das, delta_das):
        # we have eular categories, but then we also need to
        # determine whether or not we categorize a response
        eular = cls.__eular__cat(baseline_das, delta_das)

        # convert category to binary response
        return cls.NonReponses[eular]

    @staticmethod
    def __eular__cat(baseline_das, delta_das):
        # if baseline das is already low
        if baseline_das < 2.4:
            if delta_das > 1.2:
                return "good"
            return "moderate"

        # if baseline das is moderate
        elif baseline_das <= 3.7:
            if delta_das > 0.6:
                return "moderate"
            return "none"

        # if baseline das is high
        # at this point, we could just use
        # else, this is for readability
        elif baseline_das > 3.7:
            if delta_das > 1.2:
                return "moderate"
            return "none"
