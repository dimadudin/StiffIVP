# Реализация методов Рунге-Кутты для решения ОДУ #

class explicit_rk:
    def __init__(self, a, b, c):
        # Таблица Бутчера #
        self.a, self.b, self.c = a, b, c
    def __call__(self, args):
        # Таблица Бутчера #
        a, b, c = self.a, self.b, self.c
        # TODO: Реализовать #

class implicit_rk:
    def __init__(self, a, b, c):
        # Таблица Бутчера #
        self.a, self.b, self.c = a, b, c
    def __call__(self, args):
        # Таблица Бутчера #
        a, b, c = self.a, self.b, self.c
        # TODO: Реализовать #