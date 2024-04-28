import datetime
from ahrs.utils import WMM
# lattitude and longitude of Thiruvananthapuram, height in km, yyyy-mm-dd format
wmm = WMM(latitude=8.524139, longitude=76.936638, height=1000, date=datetime.date(2024, 4, 24))  
# X, Y, Z magnetic field in nanoTesla
print(wmm.magnetic_elements)
