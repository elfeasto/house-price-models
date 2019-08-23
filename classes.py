import matplotlib


class Area:
    """
    An area consists of a collection of grids
    The area is either
    (a) interior, if it represents all points
    inside any of the grids
    (b) exterior, if it represents all points
    outside any of the grids

    A rating value is assigned to each area to represent the estimated
    price increase(or decrease) associated with living in this area
    """
    def __init__(self, grids, interior=True, rating=None, color=None):
        self.grids = grids
        self.rating = rating
        self.interior = interior
        self.color = color

    def contains_house(self, house):
        """
        :param data(df): houses with long and lat attributes
        :return(bool): Whether house is in the area
        """
        if self.interior:
            # check to see if house is in any of the grids
            for grid in self.grids:
                in_grid = grid.contains_point(house["long"], house["lat"])
                if in_grid:
                    return True
            # only get to here if house is not in any grid
            return False
        else:
            # check to see if house is in any of the grids
            for grid in self.grids:
                in_grid = grid.contains_point(house["long"], house["lat"])
                if in_grid:
                    return False
            # only get to here if house is not in any grid
            return True

        pass

    def _get_houses_contained(self, houses):
        f = lambda h: self.contains_house(h)
        mask = houses.apply(func=f, axis="columns")
        houses_contained = houses[mask]
        return houses_contained

    def avg_house_price(self, data):
        area_houses = self._get_houses_contained(data)
        return area_houses["price"].mean()

    def avg_diff_value(self, data):
        area_houses = self._get_houses_contained(data)
        return area_houses["diff"].mean()

    def add_area_diff_rating(self, data):
        """
        Adds(or changes if it already exists) the
        property rating to be the average of the diff column in the data
        :param data:
        :return:
        """
        avg_diff = self.avg_diff_value(data)
        self.rating = avg_diff

    def draw(self, axis):
        """
        If area is exterior DO NOT DRAW
        :param axis:
        :return:
        """
        if self.interior:
            for grid in self.grids:
                grid.draw(axis, self.color)


class Grid:
    """
    A class to manufacture rectangle objects
    NB Corner is bottom left to fit in with matplotlib
    NB value is deprecated since value is now tied to area class
    """

    def __init__(self, corner, w, h, value=None):
        """ Initialize rectangle at corner, with width w, height h """
        self.corner = corner
        self.width = w
        self.height = h
        self.value = value

    def contains_point(self, x, y):
        x_inside = (self.corner[0] <= x) and (x < self.corner[0] + self.width)
        y_inside = (self.corner[1] <= y) and (y < self.corner[1] + self.height)
        return x_inside and y_inside

    def contains_house(self, house):
        return self.contains_point(house["long"], house["lat"])

    def draw(self, axis, color=None):
        """
        if no color is given the grid will have a blue outline
        :param axis: matplotlib axis object
        :return: None
        """
        if color :
            to_draw = matplotlib.patches.Rectangle(self.corner,
                                                   self.width, self.height,
                                                   linewidth=1, edgecolor= color,
                                                   facecolor='none')
        else:
            to_draw = matplotlib.patches.Rectangle(self.corner,
                                                   self.width, self.height,
                                                   linewidth=1, edgecolor= "b",
                                                   facecolor='none')
        axis.add_patch(to_draw)


#testing shizzle below
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from project_tools import *
    train_data, test_data = get_train_test_data()
    train_data.loc[0,"lat"] = .5
    train_data.loc[0,"long"] = .5

    my_grid1 = Grid((.5,.5),5,5)
    my_grid2 = Grid( (.25,.25),.5,.5 )
    my_grids = [my_grid1, my_grid2]
    my_area = Area(my_grids, rating = 0, color="green")
    in_grid = my_grid1.contains_house(train_data.iloc[0])
    print(in_grid)
    in_area = my_area.contains_house(train_data.iloc[0])
    print(in_area)
    not_my_area = Area(my_grids,False,10)
    my_areas = [my_area, not_my_area]
    print(my_areas)
    print(not_my_area.contains_house(train_data.iloc[0]))
    print(not_my_area.contains_house(train_data.iloc[1]))
    area_rating = area_rating_series(train_data, my_areas)
    print(area_rating)
