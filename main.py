import gurobipy as gp
from gurobipy import GRB
import data_loader

I = 8   # sectores
C = 800  # carabineros
T = 150  # días
E = 6   # especialidades

class Model:
    def __init__(self):
        self.model = gp.Model("OptimizationModel")
        self.data_loader = data_loader.DataLoader()

    def build_model(self):
        # parámetros monetarios
        self.f = self.data_loader.load_data('fi.csv', ['Sector'], 'fi')
        self.c = self.data_loader.load_data('ceit.csv', ['Especialidad', 'Sector', 'Dia'], 'ceit')
        self.s = self.data_loader.load_data('set.csv', ['Especialidad', 'Dia'], 'set')
        self.n = self.data_loader.load_data('ne.csv', ['Especialidad'], 'ne')
        self.b = self.data_loader.load_data('bi.csv', ['Sector'], 'bi')
        self.a = 100000000
        # parámetros de sector
        self.j = self.data_loader.load_data('jet.csv', ['Especialidad', 'Dia'], 'jet')
        self.k = self.data_loader.load_data('keit.csv', ['Especialidad', 'Sector', 'Dia'], 'keit')
        self.q = self.data_loader.load_data('qeit.csv', ['Especialidad', 'Sector', 'Dia'], 'qeit')
        self.u = self.data_loader.load_data('ueit.csv', ['Especialidad', 'Sector', 'Dia'], 'ueit')
        self.g = self.data_loader.load_data('geikt.csv', ['Especialidad', 'Desde', 'Hacia', 'Dia'], 'geikt')
        # parámetros de especialidad
        self.z = self.data_loader.load_data('zce.csv', ['Carabinero', 'Especialidad'], 'zce')
        self.d = 294

        # variables de decisión
        self.x = self.model.addVars(range(E), range(I), range(T), vtype=GRB.CONTINUOUS, name="x")
        self.y = self.model.addVars(range(C), range(I), range(T), vtype=GRB.BINARY, name="y")
        self.w = self.model.addVars(range(I), range(T), vtype=GRB.CONTINUOUS, name="w")
        self.V = self.model.addVars(range(E), range(I), range(I), range(T), vtype=GRB.CONTINUOUS, name="V")

        # función objetivo
        self.model.setObjective(
            gp.quicksum((self.c[e][i][t] + self.s[e][t]) * self.x[e, i, t] for e in range(E) for i in range(I) for t in range(T)) +
            gp.quicksum(self.n[e] * self.z[m][e] * self.y[m, i, t] for m in range(C) for i in range(I) for e in range(E) for t in range(T)) -
            gp.quicksum(self.w[i, t] * self.b[i] for i in range(I) for t in range(T)) + 
            gp.quicksum((self.g[e][o][i][t] * self.V[e, o, i, t]) for e in range(E) for o in range(I) for i in range(I) for t in range(T) if i != o),
            GRB.MINIMIZE
        )

        # restricciones
        M = 1e6

        # R: restricción de presupuesto
        self.restriccion_presupuesto = self.model.addConstrs(
            gp.quicksum((self.c[e][i][t] + self.s[e][t]) * self.x[e, i, t] for e in range(E) for t in range(T)) +
            gp.quicksum(self.n[e] * self.z[m][e] * self.y[m, i, t] for m in range(C) for e in range(E) for t in range(T)) + 
            gp.quicksum((self.g[e][o][i][t] * self.V[e, o, i, t]) for e in range(E) for o in range(I) for t in range(T) for i in range(I) if i != o) 
            <= self.f[i] for i in range(I)
        )

        # R: límite movilidad
        self.model.addConstrs(
            gp.quicksum(self.V[e, o, i, t] for o in range(I) if o != i) <= self.x[e, i, t]
            for e in range(E) for i in range(I) for t in range(T)
        )

        # R1: disponibilidad diaria
        self.model.addConstrs(
            gp.quicksum(self.x[e, i, t] for i in range(I)) <= self.j[e][t]
            for e in range(E) for t in range(T)
        )

        # R3: mínimo por sector
        self.model.addConstrs(
            gp.quicksum(self.x[e, i, t] for e in range(E)) >= self.k[e][i][t] + self.q[e][i][t]
            for e in range(E) for i in range(I) for t in range(T)
        )

        # R4: máximo por sector
        self.model.addConstrs(
            self.x[e, i, t] <= self.u[e][i][t]
            for e in range(E) for i in range(I) for t in range(T)
        )

        # R6: máximo días por carabinero
        self.model.addConstrs(
            gp.quicksum(self.y[m, i, t] for i in range(I) for t in range(T)) <= self.d
            for m in range(C)
        )

        # R7: relación X e Y
        self.model.addConstrs(
            gp.quicksum(self.y[m, i, t] * self.z[m][e] for m in range(C)) +
            gp.quicksum(self.V[e, o, i, t] for o in range(I) if o != i) -
            gp.quicksum(self.V[e, i, o, t] for o in range(I) if o != i)
            == self.x[e, i, t]
            for e in range(E) for i in range(I) for t in range(T)
        )

        # R8: definición carabineros extra
        self.model.addConstrs(
            self.w[i, t] == gp.quicksum(self.x[e, i, t] - self.q[e][i][t] for e in range(E))
            for i in range(I) for t in range(T)
        )

        # R9: límite bono anual
        self.model.addConstrs(
            gp.quicksum(self.w[i, t] for t in range(T)) <= self.a
            for i in range(I)
        )

    def solve_model(self):
        self.model.optimize()

    def analysis_scenarios(self):

        print("\n--- Análisis de escenarios ---")
        self.model.NumScenarios = 2

        # escenario 0: Modelo base
        self.model.Params.ScenarioNumber = 0 
        self.model.ScenNName  = "Modelo base"

        # escenario 1: Aumento de presupuesto f
        self.model.Params.ScenarioNumber  = 1
        self.model.ScenNName  = "Aumento de presupuesto f"
        for i in range(I): 
            self.restriccion_presupuesto[i].ScenNRHS = self.f[i] * 0.1  # Aumentar el RHS de la restricción de presupuesto en un 10%
        



    def sensitivity_analysis(self):
        print("\n--- Análisis de sensibilidad (modelo relajado LP) ---")
        if self.model.IsMIP:
            print("⚠️  El modelo es MIP. El análisis de sensibilidad solo está disponible para modelos LP.")
            return

        print("\nVariables continuas:")
        for v in self.model.getVars():
            if v.VType == GRB.CONTINUOUS:
                try:
                    print(f"{v.VarName}: valor óptimo = {v.X:.2f}, SAObjLow = {v.SAObjLow:.2f}, SAObjUp = {v.SAObjUp:.2f}, Reduced Cost = {v.RC:.2f}")
                except gp.GurobiError as e:
                    print(f"{v.VarName}: No disponible ({e})")

        print("\nRestricciones:")
        for r in self.model.getConstrs():
            try:
                print(f"{r.ConstrName}: RHS = {r.RHS:.2f}, Shadow Price = {r.Pi:.2f}, SARHSLow = {r.SARHSLow:.2f}, SARHSUp = {r.SARHSUp:.2f}")
            except gp.GurobiError as e:
                print(f"{r.ConstrName}: No disponible ({e})")

    def print_results(self):
        for s in range(self.model.NumScenarios):
            self.model.Params.ScenarioNumber = s
            print(f"\n\n------ Scenario {s} ({self.model.ScenNName})")

            if self.model.ModelSense * self.model.ScenNObjVal >= GRB.INFINITY:
                if self.model.ModelSense * self.model.ScenNObjBound >= GRB.INFINITY:
                    print("\nINFEASIBLE")
                else:    
                    print("\nNO SOLUTION")
            else:
                print(f"\nValor objetivo: {self.model.ModelSense * self.model.ScenNObjVal:.2f}")
                # if self.model.status == GRB.OPTIMAL:
                #     print(f"\nValor óptimo: {self.model.objVal:.2f} unidades de utilidad\n")
                #     self.sensitivity_analysis()
                # elif self.model.status == GRB.INFEASIBLE:
                #     print("Modelo infactible. Calculando IIS...")
                #     self.model.computeIIS()
                #     self.model.write("modelo.ilp")
                #     print("Archivo IIS escrito como 'modelo.ilp' en el directorio actual.")
                # elif self.model.status == GRB.INF_OR_UNBD:
                #     print("Modelo infactible o no acotado. Reintentando solo para infactibilidad...")
                #     self.model.setParam('DualReductions', 0)
                #     self.model.optimize()
                #     if self.model.status == GRB.INFEASIBLE:
                #         print("Modelo infactible tras desactivar DualReductions. Calculando IIS...")
                #         self.model.computeIIS()
                #         self.model.write("modelo.ilp")
                #         print("Archivo IIS escrito como 'modelo.ilp' en el directorio actual.")
                #     else:
                #         print("No se pudo encontrar una solución óptima ni IIS.")
                # else:
                #     print("No se pudo encontrar una solución óptima.")

def main():
    model = Model()
    model.build_model()
    model.analysis_scenarios()
    model.solve_model()
    model.print_results()

if __name__ == "__main__":
    main()