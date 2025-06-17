import gurobipy as gp
from gurobipy import GRB
import data_loader
import data_saver

I = 8   # sectores
C = 800  # carabineros
T = 150  # días
E = 6   # especialidades

class Model:
    def __init__(self):
        self.model = gp.Model("OptimizationModel")
        self.model.setParam('Timelimit', 1800)
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
        
        # R1: disponibilidad diaria
        self.disponibilidad_diaria = self.model.addConstrs(
            gp.quicksum(self.x[e, i, t] for i in range(I)) <= self.j[e][t]
            for e in range(E) for t in range( T)
        )

        # R2: restricción de presupuesto
        self.restriccion_presupuesto = self.model.addConstrs(
            gp.quicksum((self.c[e][i][t] + self.s[e][t]) * self.x[e, i, t] for e in range( E) for t in range( T)) +
            gp.quicksum(self.n[e] * self.z[m][e] * self.y[m, i, t] for m in range( C) for e in range( E) for t in range( T)) + 
            gp.quicksum((self.g[e][o][i][t] * self.V[e, o, i, t]) for e in range(E) for o in range(I) for t in range(T) for i in range(I) if i != o) 
            <= self.f[i] for i in range( I)
        )
        
        # R3: mínimo por sector
        self.model.addConstrs(
            gp.quicksum(self.x[e, i, t] for e in range(E)) >= self.k[e][i][t] + self.q[e][i][t]
            for e in range(E) for i in range(I) for t in range(T)
        )

        # R4: máximo por sector
        self.maximo_carabineros_sector = self.model.addConstrs(
            self.x[e, i, t] <= self.u[e][i][t]
            for e in range(E) for i in range(I) for t in range(T)
        )

        # R5: máximo días por carabinero
        self.model.addConstrs(
            gp.quicksum(self.y[m, i, t] for i in range(I) for t in range(T)) <= self.d
            for m in range(C)
        )

        # R6: relación X e Y
        self.model.addConstrs(
            gp.quicksum(self.y[m, i, t] * self.z[m][e] for m in range(C)) +
            gp.quicksum(self.V[e, o, i, t] for o in range(I) if o != i) -
            gp.quicksum(self.V[e, i, o, t] for o in range(I) if o != i)
            == self.x[e, i, t]
            for e in range(E) for i in range(I) for t in range(T)
        )

        # R7: definición carabineros extra
        self.model.addConstrs(
            self.w[i, t] == gp.quicksum(self.x[e, i, t] - self.q[e][i][t] for e in range(E))
            for i in range(I) for t in range(T)
        )

        # R8: límite bono anual
        self.model.addConstrs(
            gp.quicksum(self.w[i, t] for t in range(T)) <= self.a
            for i in range(I)
        )

        # R9: límite movilidad
        self.model.addConstrs(
            gp.quicksum(self.V[e, o, i, t] for o in range( I) if o != i) <= self.x[e, i, t]
            for e in range( E) for i in range( I) for t in range( T)
        )

    def solve_model(self):
        self.model.optimize()

    def analysis_scenarios(self):

        print("\n--- Análisis de escenarios ---")
        self.model.NumScenarios = 4

        # escenario 0: Modelo base
        self.model.Params.ScenarioNumber = 0 
        self.model.ScenNName  = "Modelo base"

        # escenario 1: Cambio de presupuesto f
        self.model.Params.ScenarioNumber  = 1
        self.model.ScenNName  = "Cambio de presupuesto f"

        for i in range(I): 
            self.restriccion_presupuesto[i].ScenNRHS = self.f[i] * 0.1 
        
        # escenario 2: Cambio de disponibilidad diaria carabineros
        self.model.Params.ScenarioNumber = 2
        self.model.ScenNName  = "Cambio disponibilidad diaria carabineros"

        for e in range(E):
            for t in range(T):
                self.disponibilidad_diaria[e, t].ScenNRHS = self.j[e][t] * 2.1
        
        # escenario 3: Cambio de maximo por sector 
        self.model.Params.ScenarioNumber = 3
        self.model.ScenNName  = "Cambio de maximo por sector"
        for e in range(E):
            for i in range(I):
                for t in range(T):
                    self.maximo_carabineros_sector[e, i, t].ScenNRHS = self.u[e][i][t] * 0.5

    def print_results(self):
        if self.model.NumScenarios > 0:
            self.print_analysis_results()
        else:
            self.print_normal_results()
    def print_normal_results(self):
        print("\n--- Resultados del modelo ---")
        if self.model.status == GRB.OPTIMAL:
            print(f"\nValor óptimo: {self.model.objVal:.2f} unidades de utilidad\n")
        elif self.model.status == GRB.INFEASIBLE:
            print("Modelo infactible. Calculando IIS...")
            self.model.computeIIS()
            self.model.write("modelo.ilp")
            print("Archivo IIS escrito como 'modelo.ilp' en el directorio actual.")
        else:
            print("No se pudo encontrar una solución óptima.")
        
    def print_analysis_results(self):
        print("\nResumen de escenarios\n")
        for s in range(self.model.NumScenarios):
            self.model.Params.ScenarioNumber = s
            print(f"Escenario {s} ({self.model.ScenNName})") 

            if self.model.ModelSense * self.model.ScenNObjVal >= GRB.INFINITY:
                if self.model.ModelSense * self.model.ScenNObjBound >= GRB.INFINITY:
                    print("\nINFEASIBLE")
                else:    
                    print("\nNO SOLUTION")
            else:
                print(f"\nValor objetivo: {self.model.ModelSense * self.model.ScenNObjVal:.2f}")
    
    def control_analysis(self):
        print("¿Desea realizar un análisis de sensibilidad? (s/n)") 
        while True:
            choice = input().strip().lower()
            if choice == 's':
                self.analysis_scenarios()
                break
            elif choice == 'n':
                print("Análisis de sensibilidad omitido.")
                break


def main():
    model = Model()
    model.build_model()
    model.control_analysis()
    model.solve_model()
    model.print_results()

    # saver = data_saver.DataSaver()
    # data = []
    # for v in model.model.getVars():
    #     data.append(f"{v.VarName} = {v.X}")
    # saver.save_data('results.txt', data)

if __name__ == "__main__":
    main()