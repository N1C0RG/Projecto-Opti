import gurobipy as gp
from gurobipy import GRB
import data_loader

class Model:
    def __init__(self):
        self.model = gp.Model("OptimizationModel")
        self.data_loader = data_loader.DataLoader()

    def build_model(self):
        # conjuntos
        I = 8   # sectores
        C = 50  # carabineros
        T = 10  # días
        E = 6   # especialidades

        # parámetros monetarios
        f = self.data_loader.load_data('fi.csv', ['Sector'], 'fi')
        c = self.data_loader.load_data('ceit.csv', ['Especialidad', 'Sector', 'Dia'], 'ceit')
        s = self.data_loader.load_data('set.csv', ['Especialidad', 'Dia'], 'set')
        n = self.data_loader.load_data('ne.csv', ['Especialidad'], 'ne')
        b = self.data_loader.load_data('bi.csv', ['Sector'], 'bi')
        a = 100000000

        # parámetros de sector
        j = self.data_loader.load_data('jet.csv', ['Especialidad', 'Dia'], 'jet')
        k = self.data_loader.load_data('keit.csv', ['Especialidad', 'Sector', 'Dia'], 'keit')
        q = self.data_loader.load_data('qeit.csv', ['Especialidad', 'Sector', 'Dia'], 'qeit')
        u = self.data_loader.load_data('ueit.csv', ['Especialidad', 'Sector', 'Dia'], 'ueit')
        g = self.data_loader.load_data('geikt.csv', ['Dia','Desde','Hacia','Costo'], 'geikt')
        # parámetros de especialidad
        z = self.data_loader.load_data('zce.csv', ['Carabinero', 'Especialidad'], 'zce')
        d = 40

        # variables de decisión
        x = self.model.addVars(range( E), range(I), range( T), vtype=GRB.CONTINUOUS, name="x")
        y = self.model.addVars(range( C), range( I), range( T), vtype=GRB.BINARY, name="y")
        w = self.model.addVars(range( I), range( T), vtype=GRB.CONTINUOUS, name="w")
        V = self.model.addVars(range( E), range( I), range( I), range( T), vtype=GRB.CONTINUOUS, name="V")

        # función objetivo
        self.model.setObjective(
            gp.quicksum((c[e][i][t] + s[e][t]) * x[e, i, t] for e in range( E) for i in range( I) for t in range( T)) +
            gp.quicksum(n[e] * z[m][e] * y[m, i, t] for m in range( C) for i in range( I) for e in range( E) for t in range( T)) -
            gp.quicksum(w[i, t] * b[i] for i in range( I) for t in range( T)) + 
            gp.quicksum((g[e][o][i][t] * V[e, o, i, t]) for e in E for o in I for t in T for i in I if i != o),
            GRB.MINIMIZE
        )

        # restricciones
        M = 1e6

        # R: restricción de presupuesto
        self.model.addConstrs(
            gp.quicksum((c[e][i][t] + s[e][t]) * x[e, i, t] for e in range( E) for t in range( T)) +
            gp.quicksum(n[e] * z[m][e] * y[m, i, t] for m in range( C) for e in range( E) for t in range( T)) + 
            gp.quicksum((g[e][o][i][t] * V[e, o, i, t]) for e in E for o in I for t in T for i in I if i != o) 
            <= f[i] for i in range( I)
        )

        # R: límite movilidad
        self.model.addConstrs(
            gp.quicksum(V[e, o, i, t] for o in range( I) if o != i) <= x[e, i, t]
            for e in range( E) for i in range( I) for t in range( T)
        )

        # R1: disponibilidad diaria
        self.model.addConstrs(
            gp.quicksum(x[e, i, t] for i in range(I)) <= j[e][t]
            for e in range(E) for t in range( T)
        )

        # R3: mínimo por sector
        self.model.addConstrs(
            gp.quicksum(x[e, i, t] for e in range( E)) >= k[e][i][t] + q[e][i][t]
            for e in range( E) for i in range( I) for t in range( T)
        )

        # R4: máximo por sector
        self.model.addConstrs(
            x[e, i, t] <= u[e][i][t]
            for e in range( E) for i in range( I) for t in range( T)
        )

        # R6: máximo días por carabinero
        self.model.addConstrs(
            gp.quicksum(y[m, i, t] for i in range( I) for t in range( T)) <= d
            for m in range( C)
        )

        # R7: relación X e Y
        self.model.addConstrs(
            gp.quicksum(y[m, i, t] * z[m][e] for m in range( C)) +
            gp.quicksum(V[e, o, i, t] for o in range( I) if o != i) -
            gp.quicksum(V[e, i, o, t] for o in range( I) if o != i)
            == x[e, i, t]
            for e in range( E) for i in range( I) for t in range( T)
        )

        # R8: definición carabineros extra
        self.model.addConstrs(
            w[i, t] == gp.quicksum(x[e, i, t] - q[e][i][t] for e in range( E))
            for i in range( I) for t in range( T)
        )

        # R9: límite bono anual
        self.model.addConstrs(
            gp.quicksum(w[i, t] for t in range( T)) <= a
            for i in range( I)
        )

    def solve_model(self):
        self.model.optimize()

    def print_results(self):
        if self.model.status == GRB.OPTIMAL:
            print(f"\nValor óptimo: {self.model.objVal:.2f} unidades de utilidad\n")
        else:
            print("No se pudo encontrar una solución óptima.")

def main():
    model = Model()
    model.build_model()
    model.solve_model()
    model.print_results()

if __name__ == "__main__":
    main()
