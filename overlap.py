class Rectangle:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def is_intersect(self, other):
        if self.min_x > other.max_x or self.max_x < other.min_x:
            return False
        if self.min_y > other.max_y or self.max_y < other.min_y:
            return False
        return True

    def get_insersec_region(self, other):
        if not self.is_intersect(other):
            return None
        min_x = max(self.min_x, other.min_x)
        max_x = min(self.max_x, other.max_x)
        min_y = max(self.min_y, other.min_y)
        max_y = min(self.max_y, other.max_y)

        return Rectangle(min_x, max_x, min_y, max_y)

    def __str__(self):
        return str(self.min_x) + '\t' + str(self.max_x) + '\t' + str(self.min_y) + '\t' + str(self.max_y)
