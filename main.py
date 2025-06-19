import gurobipy as gp
from gurobipy import GRB
import data_loader
import matplotlib.pyplot as plt
import data_saver
import numpy as np

class Model:
    def __init__(self):
        self.model = gp.Model("OptimizationModel")
        self.data_loader = data_loader.DataLoader()
        self.colors = ['#6FA8DC', '#FFB347', '#77DD77', '#FF6961', '#B39EB5', '#FFF380', '#FFB7B2']
        
    def build_model(self):
        # conjuntos
        self.I = 7   # sectores
        self.C = 800  # carabineros
        self.T = 365  # días
        self.E = 6   # especialidades

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
        self.x = self.model.addVars(range(self.E), range(self.I), range(self.T), vtype=GRB.CONTINUOUS, name="x")
        self.y = self.model.addVars(range(self.C), range(self.I), range(self.T), vtype=GRB.BINARY, name="y")
        self.w = self.model.addVars(range(self.I), range(self.T), vtype=GRB.CONTINUOUS, name="w")
        # función objetivo
        self.model.setObjective(
            gp.quicksum((self.c[e][i][t] + self.s[e][t]) * self.x[e, i, t] for e in range(self.E) for i in range(self.I) for t in range(self.T)) +
            gp.quicksum(self.n[e] * self.z[m][e] * self.y[m, i, t] for m in range(self.C) for i in range(self.I) for e in range(self.E) for t in range(self.T)) -
            gp.quicksum(self.w[i, t] * self.b[i] for i in range(self.I) for t in range(self.T)),
            GRB.MINIMIZE
        )

        # restricciones
        M = 1e6

        # R: restricción de presupuesto
        self.restriccion_presupuesto = self.model.addConstrs(
            gp.quicksum((self.c[e][i][t] + self.s[e][t]) * self.x[e, i, t] for e in range(self.E) for t in range(self.T)) +
            gp.quicksum(self.n[e] * self.z[m][e] * self.y[m, i, t] for m in range(self.C) for e in range(self.E) for t in range(self.T))
            <= self.f[i] for i in range(self.I)
        )

        # R1: disponibilidad diaria
        self.disponibilidad_diaria = self.model.addConstrs(
            gp.quicksum(self.x[e, i, t] for i in range(self.I)) <= self.j[e][t]
            for e in range(self.E) for t in range(self.T)
        )

        # R3: mínimo por sector
        self.model.addConstrs(
            gp.quicksum(self.x[e, i, t] for e in range(self.E)) >= self.k[e][i][t] + self.q[e][i][t]
            for e in range(self.E) for i in range(self.I) for t in range(self.T)
        )

        # R4: máximo por sector
        self.maximo_carabineros_sector = self.model.addConstrs(
            self.x[e, i, t] <= self.u[e][i][t]
            for e in range(self.E) for i in range(self.I) for t in range(self.T)
        )

        # R6: máximo días por carabinero
        self.model.addConstrs(
            gp.quicksum(self.y[m, i, t] for i in range(self.I) for t in range(self.T)) <= self.d
            for m in range(self.C)
        )

        # R7: relación X e Y
        self.model.addConstrs(
            gp.quicksum(self.y[m, i, t] * self.z[m][e] for m in range(self.C))
            == self.x[e, i, t]
            for e in range(self.E) for i in range(self.I) for t in range(self.T)
        )

        # R8: definición carabineros extra
        self.model.addConstrs(
            self.w[i, t] == gp.quicksum(self.x[e, i, t] - self.q[e][i][t] for e in range(self.E))
            for i in range(self.I) for t in range(self.T)
        )

        # R9: límite bono anual
        self.model.addConstrs(
            gp.quicksum(self.w[i, t] for t in range(self.T)) <= self.a
            for i in range(self.I)
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

        for i in range(self.I): 
            self.restriccion_presupuesto[i].ScenNRHS = self.f[i] * 2
        
        # escenario 2: Cambio de disponibilidad diaria carabineros
        self.model.Params.ScenarioNumber = 2
        self.model.ScenNName  = "Cambio disponibilidad diaria carabineros"

        for e in range(self.E):
            for t in range(self.T):
                self.disponibilidad_diaria[e, t].ScenNRHS = self.j[e][t] * 0.9
        
        # escenario 3: Cambio de maximo por sector 
        self.model.Params.ScenarioNumber = 3
        self.model.ScenNName  = "Cambio de maximo por sector"
        for e in range(self.E):
            for i in range(self.I):
                for t in range(self.T):
                    self.maximo_carabineros_sector[e, i, t].ScenNRHS = self.u[e][i][t] * 1.2

    def print_normal_results(self):
        if self.model.status == GRB.OPTIMAL:
            print(f"\nValor óptimo: {self.model.objVal:.2f} unidades de utilidad\n")
        elif self.model.status == GRB.INFEASIBLE:
            print("Modelo infactible")
        elif self.model.status == GRB.UNBOUNDED:
            print("Modelo no acotado")
        else:
            print("No se pudo encontrar una solución óptima.")
            
    def scenario_graph_controller(self):
        while True:
            print("¿Quieres graficar los resultados de los escenarios? (s/n)")
            choice = input().strip().lower()
            if choice == 's':
                self.plot_x_scenarios()
                self.plot_x_scenarios_each_sector()
                break
            elif choice == 'n':
                print("No se graficarán los resultados de los escenarios.")
                break
    
    def plot_x_scenarios(self):
        for s in range(self.model.NumScenarios):
            self.model.Params.ScenarioNumber = s
            plt.figure(figsize=(10, 6))
            for i in range(self.I): 
                y = []
                for t in range(self.T):  
                    total = sum(self.x[e, i, t].X for e in range(self.E))  
                    y.append(total)
                plt.plot(range(self.T), y, label=f"Sector {i}")
            scenario_name = getattr(self.model, "ScenNName", f"Escenario {s}")
            plt.title(f"Carabineros por sector en 1 año - {scenario_name}")
            plt.xlabel("Día")
            plt.ylabel("Number de carabineros")
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    def plot_x_scenarios_each_sector(self):
        for s in range(self.model.NumScenarios):
            self.model.Params.ScenarioNumber = s
            plt.figure(figsize=(10, 6))
            for i in range(self.I): 
                y = []
                for t in range(self.T):  
                    total = sum(self.x[e, i, t].X for e in range(self.E))  
                    y.append(total)
                plt.plot(range(self.T), y, color=self.colors[i])
                scenario_name = getattr(self.model, "ScenNName", f"Escenario {s}")
                plt.title(f"Carabineros sector {i + 1} en 1 año - {scenario_name}")
                plt.xlabel("Día")
                plt.ylabel("Number de carabineros")
                plt.legend()
                plt.tight_layout()
                plt.show()

    def print_analysis_results(self):
        print("\nResumen de escenarios\n")
        for s in range(self.model.NumScenarios):
            self.model.Params.ScenarioNumber = s
            print(f"\nEscenario {s} ({self.model.ScenNName})") 

            if self.model.ModelSense * self.model.ScenNObjVal >= GRB.INFINITY:
                if self.model.ModelSense * self.model.ScenNObjBound >= GRB.INFINITY:
                    print("Modelo no acotado")
                else:   
                    print("Modelo infactible")  
            else:
                print(f"\nValor objetivo: {self.model.ModelSense * self.model.ScenNObjVal:.2f}")
    
    def graph_results_controller(self):
        while True:
            print("¿Quieres graficar los resultados? (s/n)")
            choice = input().strip().lower()
            if choice == 's':
                self.x_graph_all_sectors_results()
                self.x_graph_specific_sector_results()
                self.w_graph_results()
                break
            elif choice == 'n':
                print("No se graficarán los resultados.")
                break

    def x_graph_all_sectors_results(self):
        for i in range(self.I):  
            y = []
            for t in range(self.T):  
                total = sum(self.x[e, i, t].X for e in range(self.E))  
                y.append(total)
            plt.plot(range(self.T), y, label=f"Sector {i + 1}")

        plt.xlabel("Día")
        plt.ylabel("Number de carabineros")
        plt.title("Carabineros por sector en 1 año")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def x_graph_specific_sector_results(self):
        for i in range(self.I):  
            y = []
            for t in range(self.T):  
                total = sum(self.x[e, i, t].X for e in range(self.E))  
                y.append(total)
            plt.plot(range(self.T), y, color=self.colors[i])
            plt.xlabel("Día")
            plt.ylabel("Number de carabineros")
            plt.title(f"Carabineros sector {i + 1} en 1 año")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def w_graph_results(self):
        for i in range(self.I):  
            y = []
            for t in range(self.T):  
                y.append(self.w[i, t].X)
            plt.plot(range(self.T), y, label=f"Sector {i + 1}")

        plt.xlabel("Día")
        plt.ylabel("Number de carabineros extra")
        plt.title("Carabineros extra por sector en 1 año")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_results(self):
        saver = data_saver.DataSaver()
        data = []
        for x in self.model.getVars():
            if x.X > 0:
                data.append(f'{x.VarName}: {x.X}')
        saver.save_data('results.txt', data)

    def control_analysis(self):
        print("¿Desea realizar un análisis de sensibilidad? (s/n)") 
        while True:
            choice = input().strip().lower()
            if choice == 's':
                self.analysis_scenarios()
                self.solve_model()
                self.print_analysis_results()
                self.scenario_graph_controller()
                break
            elif choice == 'n':
                print("Análisis de sensibilidad omitido.")
                self.solve_model()
                self.print_normal_results()
                self.save_results()
                self.graph_results_controller()
                break

def main():
    model = Model()
    model.build_model()
    model.control_analysis()
    

if __name__ == "__main__":
    main()
