from tabnanny import check
from typing import Literal, Type


def check_kwargs(arguments: dict, parameters: dict):

    argument_keys = list(arguments.keys())
    parameter_keys = list(parameters.keys())

    def check_keywords() -> Literal[True]:
        if not all([k in parameter_keys for k in argument_keys]):
            raise TypeError(
                f"Undefined parameter. Choose one of the following: {parameter_keys}"
            )
        else:
            return True

    def check_types():
        check_state = []
        for k, v in arguments.items():
            if not isinstance(v, (parameters[k] | list)):
                raise TypeError(f"{k} should be of type {parameters[k]} or a list")
            else:
                check_state.append(True)
        return all(check_state)

    def check_lists():
        check_state = []
        for k, v in arguments.items():
            if (v is not None) & (isinstance(v, list)):
                if not all([isinstance(entry, parameters[k]) for entry in v]):
                    raise TypeError(
                        f"{k} should only be of type {parameters[k]} in a list"
                    )
            else:
                check_state.append(True)
        return all(check_state)

    checks = [check_keywords(), check_types(), check_lists()]
    return all(checks)


# TODO: Check for nonnegativeness
