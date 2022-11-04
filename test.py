def rotation(t):
    h = int(t.split(":")[0]) % 12
    m = int(t.split(":")[1])
    hour_rot_permunite = (h * 60 + m) * (360 / 12 / 60)
    mun_rot_permunite = m * (360 / 60)
    rot = abs(hour_rot_permunite - mun_rot_permunite)
    return int(rot)


# print(rotation("19:05"))
from datetime import datetime

curr = datetime.now().date()
print(curr)
curr = datetime.strptime("2022-11-13", "%Y-%m-%d").date()
pre = datetime.strptime("2013-11-12", "%Y-%m-%d").date()
print((curr - pre).days)
