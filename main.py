import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import data_loader

class Model: 
    def __init__(self): 
        self.model = gp.Model("OptimizationModel")
        self.data_loader = data_loader.DataLoader()


        def load_data(self):
            pass 

        def build_model(self): 
            # conjuntos 
            I = self.data_loader.load_data() # conjunto sectores con i in I
            C = self.data_loader.load_data() # conjunto carabienros con m in C
            T = 360 # tiempo en dias 
            E = self.data_loader.load_data() # conjunto especialidades e in E 

            # parametros monetarios 
            f = self.data_loader.load_data() # presupuesto anual por sector 
            c = self.data_loader.load_data('ceit.csv', ['Sector', 'Especialidad', 'Dia'], 'ceit') # costo movilizacion a sector para i especialidad e dia t
            s = self.data_loader.load_data() # costo mantencion equipamiento para e dia t 
            n = self.data_loader.load_data() # sueldo diario carbienro especialidad e 
            b = self.data_loader.load_data() # bono por sector i 
            a = self.data_loader.load_data() # cantidad maxima bono anual 

            # parametros de sector 
            j = self.data_loader.load_data('jet.csv', ['Especialidad', 'Dia'], 'jet') # cantidad de carabienros e disponibles el dia t 
            k = self.data_loader.load_data() # cantidad minima carabienros e sector i dia t 
            q = self.data_loader.load_data() # cantidad caribienros extra e necesaria para i dia t 
            u = self.data_loader.load_data() # cantidad maxima carabienros e en sector i dia t 
            g = self.data_loader.load_data() # 1 si sector i es critico dia t 
            v = self.data_loader.load_data() # 1 si sector i necesita carabinero con especialidad e el dia t

            # parametros de especialidad
            z = self.data_loader.load_data() # 1 si carabienro e tiene especialidad e 
            d = self.data_loader.load_data() # maximo dias que un carabinero puede trabajar al año 

            # variables de decision
            x = self.model.addVars(E, I, T, vtype=GRB.CONTINUOUS, name="x") # cantidad de carabineros con especialidad e en i el dia t 
            y = self.model.addVars(C, I, T, vtype=GRB.BINARY, name="y") # 1 si el carabinero trabajo en i el dia t
            w = self.model.addVars(I, T, vtype=GRB.CONTINUOUS, name="w") # cantidad de carabineros extra en i el dia t 

            # funcion objetivo

            self.model.setObjective(
                gp.quicksum((c[e][i][t] + s[e][t]) * x[i][e][t] for i in I for e in E for t in range(T)) +
                gp.quicksum(n[e] * z[m][e]* y[i][e][t] for m in C for i in I for e in E for t in range(T)) -
                gp.quicksum(w[i][t] * b[i] for i in I for t in range(T)), GRB.MINIMIZE
            )

            # restricciones

            M = 1e6  # Big M

            # R1:  Restriccion de disponibilidad en un dia
            self.model.addConstrs(
                gp.quicksum(x[e][i][t] for e in E) <= j[i][t] for i in I for t in range(T)
            )

            # R3: Restriccion de cantidad minima de carabineros por sector
            self.model.addConstrs(
                gp.quicksum(x[e][i][t] for e in E) >= k[i][t] + q[e][i][t] * g[i][t] for e in E for i in I for t in range(T)
            )

            # R4: Compatibilidad entre requerimiento en el sector y especialidad
            self.model.addConstrs(
                x[e][i][t] <= M * v[i][e][t] for e in E for i in I for t in range(T)
            )

            # R5: Restriccion de cantidad maxima de carabineros por sector
            self.model.addConstrs(
                gp.quicksum(x[e][i][t] for e in E) <= u[i][t] for i in I for t in range(T)
            )

            # R6: Cada carabinero puede trabajar un maximo de d dıas en el año 
            self.model.addConstrs(
                gp.quicksum(y[m][i][t] for i in I for t in range(T)) <= d for m in C 
            ) 

            # R7: Relacion X e Y 
            self.model.addConstrs(
                gp.quicksum(y[m][i][t] * z[m][e] for m in C) == x[e][i][t] for e in E for i in I for t in range(T)
            ) 

            # R8: Definicion cantidad de carabineros extra
            self.model.addConstrs(
                w[i][t] == gp.quicksum(x[e][i][t] - q[e][i][t] for e in E) for i in I for t in range(T)
            )

            # R9: Restriccion limite de bono 
            self.model.addConstrs(
                gp.quicksum(w[i][t] for t in range(T)) <= a[i] for i in I
            )

        def solve_model(self): 
            self.model.optimize()

        def print_results(self):
            if self.model.status == GRB.OPTIMAL:
                print(f"\nValor óptimo: {self.model.objVal:.2f} unidades de utilidad\n")
            else:
                print("No se pudo encontrar una solucion optima.")

def main():
    model = Model()
    model.data_loader.load_data()  
    model.build_model()  
    model.solve_model()  
    model.print_results()  

if __name__ == "__main__":
    main()