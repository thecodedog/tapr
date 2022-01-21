import re

from .filtering import matches

ALCHEMY_STRING = "__ALCHEMY__"


class NTableAlchemy:
    def __init__(self, ntbl):
        self._ntbl = ntbl

    def __getitem__(self, index):
        ntbl = self._ntbl
        for k, v in index.items():
            # k should be a dimension name, v should be a regex
            findex = list(filter(matches(v), ntbl.ntable_map(k)))
            ntbl = ntbl.ntable_map(k)[findex]
            label_map = {}
            for label in ntbl.ntable_map(k):
                match = re.match(v, label)
                if len(match.groups()) == 0:
                    label_map[label] = label.replace(
                        match.string, ALCHEMY_STRING
                    )
                else:
                    new_label = label
                    for string in match.groups():
                        new_label = new_label.replace(string, ALCHEMY_STRING)
                    label_map[label] = new_label
            ntbl = ntbl.ntable_map(k).relabel(**label_map)

        return ntbl

    def __setitem__(self, index, value):
        from .ntable import NTable

        if not isinstance(value, NTable):
            raise ValueError(
                f"value must be of type NTable, not {type(value)}"
            )

        for k, v in index.items():
            my_ntable_map = self._ntbl.ntable_map(k)
            ur_ntable_map = value.ntable_map(k)
            alchemy_labels = list(
                ur_ntable_map.contains(ALCHEMY_STRING).ntable_map(k)
            )
            label_map = {
                label: label.replace(ALCHEMY_STRING, v)
                for label in alchemy_labels
            }

            for old_label, new_label in label_map.items():
                my_ntable_map[new_label] = ur_ntable_map[old_label]


class NTableMapAlchemy:
    def __init__(self, ntable_map):
        self._ntable_map = ntable_map

    def __getitem__(self, index):
        return NTableAlchemy(self._ntable_map.ntable)[
            {self._ntable_map.dim: index}
        ]

    def __setitem__(self, index, value):
        NTableAlchemy(self._ntable_map.ntable)[
            {self._ntable_map.dim: index}
        ] = value
