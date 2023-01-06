import random
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import pygame as pg
import os
import imageio

class Node(object):

    def __init__(self, name):
        ''' Se considera el nombre como un string '''
        self.name = name
        self.Posx = posx = 0
        self.Posy = posy = 0

    def get_name(self):
        return self.name

class Edge(object):

    def __init__(self, src, dest):
        '''src = source node, dest = destination node '''
        self.src = src
        self.dest = dest

    def get_source(self):
        return self.src

    def get_destination(self):
        return self.dest

class Graph(object):
    def __init__(self):
        self.nodes = []
        self.edges = {}

    def add_node(self, node):
        if node in self.nodes:
            raise ValueError('Duplicate Node')
        else:
            self.nodes.append(node)
            self.edges[node] = []

    def get_nodes(self):
      return self.nodes

    def get_edges(self):
      return self.edges

    def get_num_aristas(self):
        num = 0
        for src in self.nodes:
          for dest in self.edges[src]:
            num+=1
        return num//2

    def add_edge(self, edge):
        src = edge.get_source()
        dest = edge.get_destination()
        if not (src in self.nodes and dest in self.nodes):
            raise ValueError('Node not in graph')
        if dest not in self.edges[src]:
          self.edges[src].append(dest)

    def children_of(self, node):
        return self.edges[node]

    def has_node(self, node):
        return node in self.nodes

    def __str__(self):
        result = ''
        for src in self.nodes:
            for dest in self.edges[src]:
                result = result + src.get_name() + \
                    '--' + dest.get_name() + '\n'
        return result[:-1] # remove last newline

    def resultado(self):
        result = {}
        for src in self.nodes:
          result[src.get_name()]=[]
          for dest in self.edges[src]:
            result[src.get_name()].append(dest.get_name())
        return result

    def save_graph(self, nombreGrafo, nombre):
        file = open(f'{os.path.abspath(os.getcwd())}/{nombre}.dot', "w")
        file.write(f"graph {nombreGrafo}" + " {" + os.linesep)
        result = ''
        for src in self.nodes:
            for dest in self.edges[src]:
                result = result + src.get_name() + \
                    '--' + dest.get_name() + '\n'
        file.write(result)
        file.write('}')
        file.close()
        
    def save_tree(self, tree, nombreGrafo, nombre):
        file = open(f'{os.path.abspath(os.getcwd())}/{nombre}.dot', "w")
        file.write(f"graph {nombreGrafo}" + " {" + os.linesep)
        result = ''
        for src in tree:
            for dest in tree[src]:
                result = result + src + '--' + dest + '\n'
        file.write(result)
        file.write('}')
        file.close()
    
    def grafoMalla(self, m, n):
        k = {}
        for i in range(m):
          for j in range(n):
            k[f'n_{i}_{j}'] = Node(f'n_{i}_{j}')
            self.add_node(k[f'n_{i}_{j}'])
        for i in range(m):
          for j in range(n):
            if i >= 1:
              self.add_edge(Edge(k[f'n_{i}_{j}'],k[f'n_{i-1}_{j}']))
            if j >= 1:
              self.add_edge(Edge(k[f'n_{i}_{j}'],k[f'n_{i}_{j-1}']))
            if i == (m-1):
              if j == (n-1):
                continue
              self.add_edge(Edge(k[f'n_{i}_{j}'],k[f'n_{i}_{j+1}']))
            elif j == (n-1):
                self.add_edge(Edge(k[f'n_{i}_{j}'],k[f'n_{i+1}_{j}']))
            else:
              self.add_edge(Edge(k[f'n_{i}_{j}'],k[f'n_{i+1}_{j}']))
              self.add_edge(Edge(k[f'n_{i}_{j}'],k[f'n_{i}_{j+1}']))
        return self.resultado()

    def grafoErdosRenyi(self, n, m):
      k = {}
      for i in range(n):
        k[f'n_{i}'] = Node(f'n_{i}')
        self.add_node(k[f'n_{i}'])
      j = 0
      while j < m:
        position = [x for x in range(n)]
        p1 = random.choice(position)
        first_node = k[f'n_{p1}']
        position.remove(p1)
        p2 = random.choice(position)
        second_node = k[f'n_{p2}']
        self.add_edge(Edge(first_node,second_node))
        self.add_edge(Edge(second_node,first_node))
        j += 1
      return self.resultado()

    def grafoGilbert(self, n, p):
      k = {}
      for i in range(n):
        k[f'n_{i}'] = Node(f'n_{i}')
        self.add_node(k[f'n_{i}'])
      for i in range(n):
        for j in range(i):
          p_random = random.uniform(0,1)
          if p_random <= p:
            if i == j:
              continue
            else:
              self.add_edge(Edge(k[f'n_{i}'], k[f'n_{j}']))
              self.add_edge(Edge(k[f'n_{j}'], k[f'n_{i}']))
      return self.resultado()

    def grafoGeografico(self, n, r):
      k = {}
      for i in range(n):
        x = random.random()
        y = random.random()
        k[f'n_{i}'] = [Node(f'n_{i}'), x, y]
        self.add_node(k[f'n_{i}'][0])
      for i in range(n):
          for j in range(i):
            dist = math.sqrt((k[f'n_{j}'][1]-k[f'n_{i}'][1])**2 + (k[f'n_{j}'][2]-k[f'n_{i}'][1])**2)
            if dist <= r:
              if i == j:
                continue
              else:
                self.add_edge(Edge(k[f'n_{i}'][0], k[f'n_{j}'][0]))
                self.add_edge(Edge(k[f'n_{j}'][0], k[f'n_{i}'][0]))
      return self.resultado()    

    def grafoBarabasiAlbert(self, n, d):
      k = {}
      for i in range(n):
        k[f'n_{i}'] = Node(f'n_{i}')
        self.add_node(k[f'n_{i}'])
        for j in range(len(k)):
          if i == j:
            continue
          p = 1 - (len(self.children_of(self.get_nodes()[j])))/d
          p_random = random.uniform(0,1)
          if p_random <= p:
              self.add_edge(Edge(k[f'n_{i}'], k[f'n_{j}']))
              self.add_edge(Edge(k[f'n_{j}'], k[f'n_{i}']))
      return self.resultado()

    def grafoDorogovtsevMendes(self, n):
      while n < 3:
        n = int(input('n debe ser mayor o igual que 3: '))
      k={}
      for i in range(3):
        k[f'n_{i}'] = Node(f'n_{i}')
        self.add_node(k[f'n_{i}'])
      lista = [x for x in range(3)]
      for j in range(3):
        self.add_edge(Edge(k[f'n_{lista[j]}'], k[f'n_{lista[j-1]}']))
        self.add_edge(Edge(k[f'n_{lista[j-1]}'], k[f'n_{lista[j]}']))
      if n > 3:
        for i in range(3,n):
          l = len(k)
          k[f'n_{i}'] = Node(f'n_{i}')
          self.add_node(k[f'n_{i}'])
          node_random = self.get_nodes()[random.randint(0, l-1)]
          children = random.choice(self.children_of(node_random))
          self.add_edge(Edge(k[f'n_{i}'], node_random))
          self.add_edge(Edge(node_random, k[f'n_{i}']))
          self.add_edge(Edge(k[f'n_{i}'], children))
          self.add_edge(Edge(children, k[f'n_{i}']))
      return self.resultado()
    
    def BFS(self, s):
        visitados = {}
        for node in self.get_nodes():
            visitados[node.get_name()] = False
        visitados[s] = True
        grafo = self.resultado()
        i = 0
        L = {}
        T = {}
        L[0] = [s]
        while L[i]:
            L[i+1] = []
            for u in L[i]:
                T[u] = []
                for v in grafo[u]:
                    if visitados[v] == False:
                        visitados[v] = True
                        L[i+1].append(v)
                        T[u].append(v)
            i += 1
        return T
    
    def Bipartido(self):
        s = random.choice(g1.get_nodes()).get_name()
        visitados = {}
        for node in self.get_nodes():
            visitados[node.get_name()] = False
        visitados[s] = True
        grafo = self.resultado()
        i = 0
        L = {}
        T = {}
        L[0] = [s]
        while L[i]:
            L[i+1] = []
            for u in L[i]:
                T[u] = []
                for v in grafo[u]:
                    if visitados[v] == False:
                        visitados[v] = True
                        L[i+1].append(v)
                        T[u].append(v)
            i += 1
        Bipartido = 'Si'
        for i in L:
            for node in L[i]:
                t.append([arista not in grafo[node] for arista in L[i]])
                if False in t[-1]:
                    Bipartido = 'No'
        return Bipartido
    
    def DFS_R(self, s):
        visitados = {}
        for node in self.get_nodes():
            visitados[node.get_name()] = False
        T = {}
        grafo = self.resultado()
        def tree_DFS(grafo, s): 
            visitados[s] = True
            T[s] = []
            for v in grafo[s]:
                if visitados[v] == False:
                    T[s].append(v)
                    tree_DFS(grafo,v)
            return T
        tree_DFS(grafo, s)
        return T
    
    
    def DFS_I(self, s):
        visitados = {}
        for node in self.get_nodes():
            visitados[node.get_name()] = False
        T = {}
        stack = []
        stack.append([s])
        path = []
        grafo = self.resultado()
        for node in grafo:
            T[node] = []
        while stack:
            v = stack[-1][-1]
            stack.pop()
            if visitados[v]:
                continue
            path.append(v)
            visitados[v] = True
            for node in grafo[v]:
                #if node not in stack:
                stack.append([v,node])
            j=-1
            while True:
                try:
                    if not(visitados[stack[j][1]]):
                        T[stack[j][0]].append(stack[j][1])
                        break
                    else:
                        stack.pop()
                    if len(stack) == 0:
                        break
                except:
                    pass
        return T

    def distancias(self):
        """Asigna pesos a los nodos del grafo de forma aleatoria
        """
        distancia = {}
        dist =[]
        for src in self.nodes:
            distancia[src.get_name()] = {}
            for dest in self.edges[src]:
                try:
                    if (src.get_name() in distancia[dest.get_name()]):
                        distancia[src.get_name()][dest.get_name()] = distancia[dest.get_name()][src.get_name()]
                except:
                    val = random.randint(1,10)
                    distancia[src.get_name()][dest.get_name()] = val
                    dist.append([[src.get_name(), dest.get_name()], val])
        return distancia, dist

    def Dijkstra(self, start):
        """ Calcula el árbol de caminos más cortos desde un nodo fuente
        Args: Nombre del nodo fuente
        Returns: Arbol con las distancias más cortas
        """
        # Asignación de las distancias a las aristas de forma aleatoria
        distancia, dist = self.distancias()

        inexplorados = [node.get_name() for node in self.nodes]
        q = {} # cola de prioridades
        s = {} # conjunto de nodos explorados
        t = {} # arbol
        for node in inexplorados:
            t[node] = []
        max_value = math.inf
        for node in self.nodes:
            q[node.get_name()] = max_value
        q[start] = 0
        
        while inexplorados:
            current_min_node = None
            for node in inexplorados:
                if current_min_node == None:
                    current_min_node = node
                elif q[node] < q[current_min_node]:
                    current_min_node = node
            grafo = self.resultado()
            neighbors = grafo[current_min_node]
            for neighbor in neighbors:
                value = q[current_min_node] + distancia[current_min_node][neighbor]
                if value < q[neighbor]:
                    q[neighbor] = value
                    s[neighbor] = current_min_node
                    t[current_min_node].append(neighbor)
                    t[neighbor].append(current_min_node)
            inexplorados.remove(current_min_node)
        arbol = {}
        for node in t:
            arbol[f'{node}_({q[node]})'] = []
            for dest in t[node]:
                arbol[f'{node}_({q[node]})'].append(f'{dest}_({q[dest]})')
        return arbol

    def find(self, parent, i):
        """ Localiza el conjunto de un elemento i
        """
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    def union(self, parent, rank, x, y):
        
        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
 
        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[y] = x
            rank[x] += 1

    def KruskalD(self):
        """ Calcula el árbol de expansión mínima
        Args:
        Returns: Arbol con las distancias más cortas
        """
        # Asignación de las distancias a las aristas de forma aleatoria
        distancia, dist = self.distancias()
        dist.sort(key = lambda edge: edge[1], reverse = False)
        t = {} # arbol
        parent = {}
        rank = {}

        for node in self.nodes:
            parent[node.get_name()] = node.get_name()
            t[node.get_name()] = []
            rank[node.get_name()] = 0

        e = 0
        i = 0
        minimumCost = 0
        while e < len(distancia)-1:
            try:
                (u, v), w = dist[i]
                i += 1
                x = self.find(parent, u)
                y = self.find(parent, v)
                if x != y:
                    e += 1
                    t[u].append(v)
                    t[v].append(u)
                    minimumCost += w
                    self.union(parent, rank, x, y)
            except:
                e += 1

        return distancia,t, minimumCost

    def KruskalI(self):
        """ Calcula el árbol de expansión mínima
        Args:
        Returns: Arbol con las distancias más cortas
        """
        # Asignación de las distancias a las aristas de forma aleatoria
        distancia, dist = self.distancias()
        dist.sort(key = lambda edge: edge[1], reverse = True)

        t = self.resultado() #arbol
        minimumCost = 0

        def DFS(grafo, s):
            visitados = {}
            for node in grafo:
                visitados[node] = False
            T = {}
            def tree_DFS(grafo, s): 
                visitados[s] = True
                T[s] = []
                for v in grafo[s]:
                    if visitados[v] == False:
                        T[s].append(v)
                        tree_DFS(grafo,v)
                return T
            tree_DFS(grafo, s)
            return T
        for i in range(len(dist)):
            (u, v), w = dist[i]

            t[u].remove(v)
            t[v].remove(u)

            visitados = DFS(t, u)
            if len(visitados) != len(self.get_nodes()):
                t[u].append(v)
                t[v].append(u)
                minimumCost += w

        return distancia, t, minimumCost

    def Prim(self):
        distancia, dist = self.distancias()
        selected_node = {}
        for node in self.get_nodes():
            selected_node[node.get_name()] = False
        #selección de un nodo inicial de forma aleatoria
        selected_node[random.choice(self.get_nodes()).get_name()] = True
        i = 0
        t = {}
        for node in self.get_nodes():
            t[node.get_name()] = []

        N = len(self.get_nodes())
        minimumCost = 0
        while i < N-1:
            minimum = math.inf
            for j in self.get_nodes():
                if selected_node[j.get_name()]:
                    for k in distancia[j.get_name()]:  
                        if (not selected_node[k]): 
                            if minimum > distancia[j.get_name()][k]:
                                minimum = distancia[j.get_name()][k]
                                u = j.get_name()
                                v = k
            t[u].append(v)
            t[v].append(u)
            minimumCost += distancia[u][v]
            selected_node[v] = True
            i +=1
        return distancia, t, minimumCost
    
def posiciones_aleatorias(a,anc,alt):
    grf=copy.deepcopy(a)
    for node in grf.get_nodes():
        node.Posx=np.random.randint(0,anc)
        node.Posy=np.random.randint(0,alt)
    return grf
    
class Animation:
    def __init__(self,grf, nom, tipo, delay):
        self._running = True
        self._display_surf = None
        self.size = self.weight, self.height = int(1920*.9), int(1080*.9)
        self.Grf=grf
        self.nom = nom
        self.tipo = tipo
        self.delay = delay
    
    def on_init(self):
        pg.init()
        self._display_surf = pg.display.set_mode(self.size, pg.HWSURFACE | pg.DOUBLEBUF)
        self._running = True
        
        
        col_texto=(30,30,30)
        col_fondo=(250,250,250)
        inicio_ancho_text=int(self.weight*.1+70)
        self.inicio_ancho_text=int(self.weight*.1+70)
        self.col_texto=(30,30,30)
        self.col_fondo=col_fondo
        
        self._display_surf.fill(col_fondo) 
        
        #Creamos fuente, de titulo y de texto:
        self.font_tit = pg.font.Font('/home/victor/Documentos/MAESTRIA_EN_CIENCIAS_DE_LA_COMPUTACION/B22/DISEÑO Y ANÁLISIS DE ALGORITMOS/Proyecto 06/century.ttf', 20)
        self.font_nor = pg.font.Font('/home/victor/Documentos/MAESTRIA_EN_CIENCIAS_DE_LA_COMPUTACION/B22/DISEÑO Y ANÁLISIS DE ALGORITMOS/Proyecto 06/century.ttf', 17)
        self.font_mini = pg.font.Font('/home/victor/Documentos/MAESTRIA_EN_CIENCIAS_DE_LA_COMPUTACION/B22/DISEÑO Y ANÁLISIS DE ALGORITMOS/Proyecto 06/century.ttf', 14)

        texto = self.font_nor.render('Numero de nodos:', True, col_texto)
        self._display_surf.blit(texto, (inicio_ancho_text, 10)) #texto
        texto = self.font_nor.render('Numero de aristas:', True, col_texto)
        self._display_surf.blit(texto, (inicio_ancho_text+380, 10)) #texto
        texto = self.font_nor.render('Tipo:', True, col_texto)
        self._display_surf.blit(texto, (inicio_ancho_text+750, 10)) #texto
        
        self.c1=1
        self.c2=1
        self.c3=1
        self.c4=1
        
        self.c=.3
        self.temp=self.weight*.7*self.height*.8*.01
        self.temp=self.weight*.7*.01
        self.cool=.99
        
        self.res=3
        self.theta=0.3
                
        #Valores de la cantidad de vertices y de aristas
        text_=str(len(self.Grf.get_nodes()))
        texto = self.font_nor.render(text_, True, col_texto,col_fondo)
        self._display_surf.blit(texto, (inicio_ancho_text+180, 10)) #texto
        text_=str(self.Grf.get_num_aristas())
        texto = self.font_nor.render(text_, True, col_texto,col_fondo)
        self._display_surf.blit(texto, (inicio_ancho_text+580, 10)) #texto
        text_ = self.nom
        texto = self.font_nor.render(text_, True, col_texto,col_fondo)
        self._display_surf.blit(texto, (inicio_ancho_text+850, 10)) #texto


        self.Grf=posiciones_aleatorias(self.Grf,self.weight*.9,self.height*.9)        
        self.cuenta_imag=0
        
        pg.display.set_caption('Visualización Grafo')
        
    def on_event(self, event):
        if event.type == pg.QUIT:
            self._running = False
            
    def on_loop(self):
        ####################################################################################
        ####################################################################################
        ###################################################################################
        pg.time.delay(self.delay)
        surface= pg.Surface((32,32))
        surface.fill(self.col_fondo)
        surface = pg.transform.scale(surface, (int(self.weight*.9), int(self.height*.9)))
        if self.tipo == 'spring':
            self.Grf=resortes(self.Grf,self.c1,self.c2,self.c3,self.c4,self.weight*.9,self.height*.9)
        elif self.tipo == 'FR':
            self.Grf = Fruchterman_Reigold(self.Grf,self.c,self.temp,self.cool,self.weight*.9,self.height*.9)
        elif self.tipo == 'quad':
            self.Grf = Fru_quad(self.Grf,self.c,self.temp,self.cool,self.weight*.9,self.height*.9,self.res,self.theta)

        for node in self.Grf.get_nodes():
            pg.draw.circle(surface,(30,30,30),(node.Posx,node.Posy),6) 
            
        for src in self.Grf.nodes:
            ps1 = (src.Posx, src.Posy)
            for dest in self.Grf.edges[src]:
                ps2 = (dest.Posx, dest.Posy)
                pg.draw.line(surface,(30,30,30),ps1,ps2) 
        
        self._display_surf.blit(surface, (50, 50))
         
        pass
    def on_render(self):
        pg.display.flip()
        pass
    def on_cleanup(self):
        pg.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False
     
        while (self._running):
            for event in pg.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()        
        self.on_cleanup()

def resortes(a,c1,c2,c3,c4,anc,alt):
    grf=copy.deepcopy(a)
    
    #calcular fuerzas en cada vértice
    vec_fuerzas={node.get_name() :(0,0) for node in grf.get_nodes()}
    
    #fuerzas de respulsión, por otros vértices
    for nodei in grf.get_nodes():
        for nodej in grf.get_nodes():
            if nodei!=nodej:
                #Distancia y angulo
                ps1=(nodei.Posx,nodei.Posy)
                ps2=(nodej.Posx,nodej.Posy)
                dis, ang = dis_ang(ps1[0],ps1[1],ps2[0],ps2[1])
                #fuerza
                if dis>0:
                    fuerza=c3/math.sqrt(dis)
                else:
                    fuerza=c3*5
                #agregamos al vec_fuerzas
                fue_f, ang_f=suma_vec(fuerza,ang,vec_fuerzas[nodei.get_name()][0],vec_fuerzas[nodei.get_name()][1])
                vec_fuerzas[nodei.get_name()]=(fue_f,ang_f)

    #fuerzas de atracción, por las aristas
    for src in grf.get_nodes():
        ps1 = (src.Posx, src.Posy)
        for dest in grf.edges[src]:
            ps2 = (dest.Posx, dest.Posy)
            dis, ang = dis_ang(ps1[0],ps1[1],ps2[0],ps2[1])
            if dis>0:
                fuerza=c1*np.log(dis/c2)
            else:
                fuerza=0
            fue_f, ang_f=suma_vec(fuerza,ang+math.pi,vec_fuerzas[src.get_name()][0],vec_fuerzas[src.get_name()][1])
            vec_fuerzas[src.get_name()]=(fue_f,ang_f)
            fue_f, ang_f=suma_vec(fuerza,ang,vec_fuerzas[dest.get_name()][0],vec_fuerzas[dest.get_name()][1])
            vec_fuerzas[dest.get_name()]=(fue_f,ang_f)
        
    #__________________ Gran atractor
    psx_GA=anc/2
    psy_GA=alt/2
    for node in grf.get_nodes():
        ps1=(node.Posx,node.Posy)
        dis, ang = dis_ang(ps1[0],ps1[1],psx_GA,psy_GA)
        if dis>0:
            #fuerza=c1*np.log(dis/c2)
            fuerza=dis*0
        else:
            fuerza=0
        fue_f, ang_f=suma_vec(fuerza,ang+math.pi,vec_fuerzas[node.get_name()][0],vec_fuerzas[node.get_name()][1])
        vec_fuerzas[node.get_name()]=(fue_f,ang_f)
        
    #Actualizar las posiciones x, y de cada vértice
    for node in grf.get_nodes():
        co_x=c4*vec_fuerzas[node.get_name()][0]*math.cos(vec_fuerzas[node.get_name()][1])
        co_y=c4*vec_fuerzas[node.get_name()][0]*math.sin(vec_fuerzas[node.get_name()][1])
        node.Posx=node.Posx+co_x
        node.Posy=node.Posy+co_y
        if node.Posx>=anc:
            node.Posx=anc-1
        elif node.Posx<1:
            node.Posx=1
        if node.Posy>=alt:
            node.Posy=alt-1
        elif node.Posy<1:
            node.Posy=1
    
    return grf    

def dis_ang(ax,ay,bx,by):
    dis=math.sqrt((ax-bx)**2+(ay-by)**2)
    ang=math.atan2( ay-by,ax-bx,)    
    return dis,ang

def suma_vec(f1,ang1,f2,ang2):
    co_x_1=f1*math.cos(ang1)
    co_y_1=f1*math.sin(ang1)
    co_x_2=f2*math.cos(ang2)
    co_y_2=f2*math.sin(ang2)
    co_x=co_x_1+co_x_2
    co_y=co_y_1+co_y_2
    f,ang=dis_ang(co_x,co_y,0,0)
    return f,ang

def Fruchterman_Reigold(a,c,temp,cool,anc,alt):
    grf=copy.deepcopy(a)
    n_vert=len(grf.get_nodes())
    area=anc*alt
    k=c*np.sqrt(area/n_vert)
    
    vec_fuerzas={node.get_name() :(0,0) for node in grf.get_nodes()}
    
    #fuerzas de repulsión por otros nodos
    for nodei in grf.get_nodes():
        for nodej in grf.get_nodes():
            if nodei!=nodej:
                #Distancia y angulo
                ps1=(nodei.Posx,nodei.Posy)
                ps2=(nodej.Posx,nodej.Posy)
                dis, ang = dis_ang(ps1[0],ps1[1],ps2[0],ps2[1])
                #fuerza
                if dis>0:
                    fuerza=k**2/dis
                else:
                    fuerza=k**2
                #agregamos al vec_fuerzas
                fue_f, ang_f=suma_vec(fuerza,ang,vec_fuerzas[nodei.get_name()][0],vec_fuerzas[nodei.get_name()][1])
                vec_fuerzas[nodei.get_name()]=(fue_f,ang_f)
                
    #fuerzas de atracción, por las aristas           
    for src in grf.get_nodes():
        ps1 = (src.Posx, src.Posy)
        for dest in grf.edges[src]:
            ps2 = (dest.Posx, dest.Posy)
            dis, ang = dis_ang(ps1[0],ps1[1],ps2[0],ps2[1])
            if dis>0:
                fuerza=(dis**2)/ k
            else:
                fuerza=0
            fue_f, ang_f=suma_vec(fuerza,ang+math.pi,vec_fuerzas[src.get_name()][0],vec_fuerzas[src.get_name()][1])
            vec_fuerzas[src.get_name()]=(fue_f,ang_f)
            fue_f, ang_f=suma_vec(fuerza,ang,vec_fuerzas[dest.get_name()][0],vec_fuerzas[dest.get_name()][1])
            vec_fuerzas[dest.get_name()]=(fue_f,ang_f)
    
    #Actualizar las posiciones x, y de cada vértice
    for node in grf.get_nodes():
        co_x=min(vec_fuerzas[node.get_name()][0], temp)*math.cos(vec_fuerzas[node.get_name()][1])
        co_y=min(vec_fuerzas[node.get_name()][0], temp)*math.sin(vec_fuerzas[node.get_name()][1])
        node.Posx=node.Posx+co_x
        node.Posy=node.Posy+co_y
        aleat=random.randint(1, 2)
        if node.Posx>=anc:
            node.Posx=anc-aleat
        elif node.Posx<1:
            node.Posx=aleat
        if node.Posy>=alt:
            node.Posy=alt-aleat
        elif node.Posy<1:
            node.Posy=aleat
    temp = cool*temp
    return grf

class quad:
    def __init__(self,bor_x,bor_y):
        self.Centro=((bor_x[0]+bor_x[1])/2,(bor_y[0]+bor_y[1])/2)
        self.Centro_masa=0
        self.Cant_masa=0
        self.QAd3=[[],[],[],[]]
        self.Bor_x=bor_x
        self.Bor_y=bor_y
        self.Vec_ver=[]
    def Agrega_Masa(self):
        self.Cant_masa+=1
    def Calcula_cen_mas(self):
        sum_x=0
        sum_y=0
        for i in range(len(self.Vec_ver)):
            sum_x+=self.Vec_ver[i][0]
            sum_y+=self.Vec_ver[i][1]
        if self.Cant_masa>0:
            self.Centro_masa=(sum_x/self.Cant_masa,sum_y/self.Cant_masa)
        else:
            self.Centro_masa=self.Centro

def Partir(cuad,res):
    cuad.QAd3[0]=quad((cuad.Bor_x[0],cuad.Centro[0]),(cuad.Bor_y[0],cuad.Centro[1]))
    cuad.QAd3[1]=quad((cuad.Centro[0],cuad.Bor_x[1]),(cuad.Bor_y[0],cuad.Centro[1]))
    cuad.QAd3[2]=quad((cuad.Bor_x[0],cuad.Centro[0]),(cuad.Centro[1],cuad.Bor_y[1]))
    cuad.QAd3[3]=quad((cuad.Centro[0],cuad.Bor_x[1]),(cuad.Centro[1],cuad.Bor_y[1]))
    
    for i in range(len(cuad.Vec_ver)):
        if cuad.Vec_ver[i][0]<cuad.Centro[0]: 
            if cuad.Vec_ver[i][1]>cuad.Centro[1]: 
                cuad.QAd3[0].Agrega_Masa()
                cuad.QAd3[0].Vec_ver.append(cuad.Vec_ver[i])
            else:       
                cuad.QAd3[2].Agrega_Masa()
                cuad.QAd3[2].Vec_ver.append(cuad.Vec_ver[i])
        else:         
            if cuad.Vec_ver[i][1]>cuad.Centro[1]:  
                cuad.QAd3[1].Agrega_Masa()
                cuad.QAd3[1].Vec_ver.append(cuad.Vec_ver[i])
            else:    
                cuad.QAd3[3].Agrega_Masa()
                cuad.QAd3[3].Vec_ver.append(cuad.Vec_ver[i])
    cuad.QAd3[0].Calcula_cen_mas()
    cuad.QAd3[1].Calcula_cen_mas()
    cuad.QAd3[2].Calcula_cen_mas()
    cuad.QAd3[3].Calcula_cen_mas()
    if cuad.QAd3[0].Cant_masa>res:
        cuad.QAd3[0]=Partir(cuad.QAd3[0], res)
    if cuad.QAd3[1].Cant_masa>res:
        cuad.QAd3[1]=Partir(cuad.QAd3[1], res)
    if cuad.QAd3[2].Cant_masa>res:
        cuad.QAd3[2]=Partir(cuad.QAd3[2], res)
    if cuad.QAd3[3].Cant_masa>res:
        cuad.QAd3[3]=Partir(cuad.QAd3[3], res)
    return cuad


def zoom(theta_lim,k,ps1,cuad,val_f):
    if cuad.Cant_masa==0:
        return val_f
    ps2=(cuad.Centro_masa[0],cuad.Centro_masa[1])
    dis, ang = dis_ang(ps1[0],ps1[1],ps2[0],ps2[1])
    if dis>0:
        theta=(cuad.Bor_x[1]-cuad.Bor_x[0])/dis
    else:
        return val_f 
    
    if theta<theta_lim:   
        fuerza=cuad.Cant_masa*k**2/dis
        fue_f, ang_f=suma_vec(fuerza,ang,val_f[0],val_f[1])
        val_f=(fue_f,ang_f)
    else:  
        if cuad.QAd3[0] != []:
            val_f=zoom(theta_lim,k,ps1,cuad.QAd3[0],val_f)
        if cuad.QAd3[1] != []:
            val_f=zoom(theta_lim,k,ps1,cuad.QAd3[1],val_f)        
        if cuad.QAd3[2] != []:
            val_f=zoom(theta_lim,k,ps1,cuad.QAd3[2],val_f)    
        if cuad.QAd3[3] != []:
            val_f=zoom(theta_lim,k,ps1,cuad.QAd3[3],val_f)    
    return val_f

def Fru_quad(a,c,temp,cool,anc,alt, res,theta_lim):
    grf=copy.deepcopy(a)
    cuad=quad((0,anc),(0,alt))
    for node in grf.get_nodes():
        cuad.Vec_ver.append([node.Posx,node.Posy])
        cuad.Agrega_Masa()
    cuad.Calcula_cen_mas()
    if cuad.Cant_masa>res:
        cuad=Partir(cuad,res)
    n_vert=len(grf.get_nodes())
    area=anc*alt
    k=c*np.sqrt(area/n_vert)
    vec_fuerzas={node.get_name() :(0,0) for node in grf.get_nodes()}

    #fuerzas de repulsión, por otros vértices, por quad
    for node in grf.get_nodes():
        ps1 = (node.Posx, node.Posy)
        val_f=(0,0)
        val_f=zoom(theta_lim,k,ps1,cuad,val_f)
        vec_fuerzas[node.get_name()]=val_f
        
    #fuerzas de atracción, por las aristas    
    for src in grf.get_nodes():
        ps1 = (src.Posx, src.Posy)
        for dest in grf.edges[src]:
            ps2 = (dest.Posx, dest.Posy)
            dis, ang = dis_ang(ps1[0],ps1[1],ps2[0],ps2[1])
            if dis>0:
                fuerza=(dis**2)/ k
            else:
                fuerza=0
            fue_f, ang_f=suma_vec(fuerza,ang+math.pi,vec_fuerzas[src.get_name()][0],vec_fuerzas[src.get_name()][1])
            vec_fuerzas[src.get_name()]=(fue_f,ang_f)
            fue_f, ang_f=suma_vec(fuerza,ang,vec_fuerzas[dest.get_name()][0],vec_fuerzas[dest.get_name()][1])
            vec_fuerzas[dest.get_name()]=(fue_f,ang_f)
    
    
    #Actualizar las posiciones x, y de cada vértice, limitado por la temperatura
    for node in grf.get_nodes():
        co_x=min(vec_fuerzas[node.get_name()][0], temp)*math.cos(vec_fuerzas[node.get_name()][1])
        co_y=min(vec_fuerzas[node.get_name()][0], temp)*math.sin(vec_fuerzas[node.get_name()][1])
        node.Posx=node.Posx+co_x
        node.Posy=node.Posy+co_y
        aleat=random.randint(1, 2)
        if node.Posx>=anc:
            node.Posx=anc-aleat
        elif node.Posx<1:
            node.Posx=aleat
        if node.Posy>=alt:
            node.Posy=alt-aleat
        elif node.Posy<1:
            node.Posy=aleat
    temp = cool*temp
    return grf


    
g = Graph()
#g.grafoMalla(25,20)
#g.grafoErdosRenyi(500,1500)
#g.grafoGilbert(500,.01)
#g.grafoGeografico(500,.1)
#g.grafoBarabasiAlbert(500,3)
g.grafoDorogovtsevMendes(500)
__name__='__main__'

# tipos: 'spring', 'FR', 'quad'

if __name__ == "__main__" :
    theApp = Animation(g, 'DorogovtsevMendes', 'quad',5)
    theApp.on_execute()
