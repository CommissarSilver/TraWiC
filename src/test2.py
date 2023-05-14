from utils import santa

coder = santa.SantaCoder()

prefix = """    def setRoomInfo(self):
        self.room_name.setText('{}({})'.format(self.data['naturalName'], self.data['roomName']))
        self.description.setText("<a style='color:#BCBCBC'>{}</a>".format(self.data['description']))
        timeStamp = int(self.data['creationDate']) / 1000
        timeArray = time.localtime(timeStamp)
        otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
        self.create_time.setText("<a style='color:#BCBCBC'>{}</a>".format(otherStyleTime))
        members = """

suffix = """ + len(self.data['admins']) + len(self.data['members'])
        memberCounter = "<a style='color:#BCBCBC'>{}/{}</a>".format(members, ('∞' if self.data['maxUsers']==0 else self.data['maxUsers']))
        self.member.setText(memberCounter)"""

middle = coder.infill((prefix, suffix),temperature=0.2)

print("\033[92m" + prefix + "\033[93m" + middle + "\033[92m" + suffix)
"""
    def setRoomInfo(self):
        self.room_name.setText('{}({})'.format(self.data['naturalName'], self.data['roomName']))
        self.description.setText("<a style='color:#BCBCBC'>{}</a>".format(self.data['description']))
        timeStamp = int(self.data['creationDate']) / 1000
        timeArray = time.localtime(timeStamp)
        otherStyleTime = time.strftime("%Y-%m-%d", timeArray)
        self.create_time.setText("<a style='color:#BCBCBC'>{}</a>".format(otherStyleTime))
        members = len(self.data['owners']) + len(self.data['admins']) + len(self.data['members'])
        memberCounter = "<a style='color:#BCBCBC'>{}/{}</a>".format(members, ('∞' if self.data['maxUsers']==0 else self.data['maxUsers']))
        self.member.setText(memberCounter)
"""