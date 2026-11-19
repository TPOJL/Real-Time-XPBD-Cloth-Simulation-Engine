# Copyright 2022 Matthias Müller - Ten Minute Physics (Port w języku Python pliku 14-cloth.html)
# Symulacja tkaniny metodą XPBD (Extended Position-Based Dynamics)
# Dodatkowe funkcjonalności:
#   1) Wiatr aerodynamiczny (Wind)
#   2) Kolizja z kulą 3D podlegającą grawitacji, którą można chwytać i rzucać myszką (Sphere collision)
#   3) Rozrywanie tkaniny metodą Dynamic Vertex Splitting (Tear cloth)

import math
import os
import re
import sys
import time
import numpy as np
from numba import njit
from PyQt5 import QtCore, QtGui, QtWidgets


# ==============================================================================
# 1. OPERACJE WEKTOROWE NA PŁASKICH TABLICACH (1:1 z 14-cloth.html)
# Wszystkie wektory 3D są przechowywane w jednej ciągłej tablicy: dla cząstki `i`
# jej współrzędne (X, Y, Z) znajdują się pod indeksami `3*i`, `3*i + 1`, `3*i + 2`.
# ==============================================================================

@njit
def vecSetZero(a, anr):
    """Zeruje wektor 3D o indeksie anr: a[anr] = (0, 0, 0)."""
    anr *= 3
    a[anr] = 0.0
    a[anr + 1] = 0.0
    a[anr + 2] = 0.0


@njit
def vecScale(a, anr, scale):
    """Mnoży wektor 3D a[anr] przez skalar scale."""
    anr *= 3
    a[anr] *= scale
    a[anr + 1] *= scale
    a[anr + 2] *= scale


@njit
def vecCopy(a, anr, b, bnr):
    """Kopiuje wektor 3D b[bnr] do wektora a[anr]."""
    anr *= 3
    bnr *= 3
    a[anr] = b[bnr]
    a[anr + 1] = b[bnr + 1]
    a[anr + 2] = b[bnr + 2]


@njit
def vecAdd(a, anr, b, bnr, scale=1.0):
    """Dodaje przeskalowany wektor: a[anr] += b[bnr] * scale."""
    anr *= 3
    bnr *= 3
    a[anr] += b[bnr] * scale
    a[anr + 1] += b[bnr + 1] * scale
    a[anr + 2] += b[bnr + 2] * scale


@njit
def vecSetDiff(dst, dnr, a, anr, b, bnr, scale=1.0):
    """Oblicza różnicę wektorów: dst[dnr] = (a[anr] - b[bnr]) * scale."""
    dnr *= 3
    anr *= 3
    bnr *= 3
    dst[dnr] = (a[anr] - b[bnr]) * scale
    dst[dnr + 1] = (a[anr + 1] - b[bnr + 1]) * scale
    dst[dnr + 2] = (a[anr + 2] - b[bnr + 2]) * scale


@njit
def vecLengthSquared(a, anr):
    """Zwraca kwadrat długości wektora ||a[anr]||^2."""
    anr *= 3
    a0, a1, a2 = a[anr], a[anr + 1], a[anr + 2]
    return a0 * a0 + a1 * a1 + a2 * a2


@njit
def vecDistSquared(a, anr, b, bnr):
    """Zwraca kwadrat odległości między dwoma punktami ||a[anr] - b[bnr]||^2."""
    anr *= 3
    bnr *= 3
    a0 = a[anr] - b[bnr]
    a1 = a[anr + 1] - b[bnr + 1]
    a2 = a[anr + 2] - b[bnr + 2]
    return a0 * a0 + a1 * a1 + a2 * a2


@njit
def vecDot(a, anr, b, bnr):
    """Iloczyn skalarny (dot product) wektorów a[anr] i b[bnr]."""
    anr *= 3
    bnr *= 3
    return a[anr] * b[bnr] + a[anr + 1] * b[bnr + 1] + a[anr + 2] * b[bnr + 2]


@njit
def vecSetCross(a, anr, b, bnr, c, cnr):
    """Iloczyn wektorowy (cross product): a[anr] = b[bnr] x c[cnr]."""
    anr *= 3
    bnr *= 3
    cnr *= 3
    a[anr] = b[bnr + 1] * c[cnr + 2] - b[bnr + 2] * c[cnr + 1]
    a[anr + 1] = b[bnr + 2] * c[cnr + 0] - b[bnr + 0] * c[cnr + 2]
    a[anr + 2] = b[bnr + 0] * c[cnr + 1] - b[bnr + 1] * c[cnr + 0]


# ==============================================================================
# 2. GLOBALNA KONFIGURACJA SCENY FIZYCZNEJ (gPhysicsScene)
# ==============================================================================

gPhysicsScene = {
    "gravity": np.array([0.0, -10.0, 0.0], dtype=np.float32),   # Przyspieszenie ziemskie (w dół osi Y)
    "dt": 1.0 / 60.0,                                           # Krok czasowy jednej klatki (60 FPS)
    "numSubsteps": 15,                                          # Liczba podkroków (substeps) na jedną klatkę
    "paused": True,                                             # Flaga pauzy symulacji na starcie
    "showEdges": False,                                         # Tryb wyświetlania samej siatki krawędzi
    "windEnabled": False,                                       # Dodatkowa funkcja 1: Wiatr
    "sphereEnabled": False,                                     # Dodatkowa funkcja 2: Kula 3D do kolizji
    "sphereGrabbed": False,                                     # Czy kula jest aktualnie trzymana myszką
    "spherePos": np.array([0.0, 0.45, 0.28], dtype=np.float32), # Pozycja środka kuli (X, Y, Z)
    "sphereVel": np.array([0.0, 0.0, 0.0], dtype=np.float32),   # Prędkość kuli (podlega grawitacji i rzutom myszką)
    "sphereRadius": 0.12,                                       # Promień kuli
    "tearEnabled": False,                                       # Dodatkowa funkcja 3: Rozrywanie tkaniny (Vertex Splitting)
    "tearThreshold": 2.00,                                      # Próg rozrywania: powyżej 2.0x długości spoczynkowej
    "objects": [],
}


def onShowEdges():
    """Przełącza widok pomiędzy pełną siatką trójkątów a siatką krawędzi (wireframe)."""
    gPhysicsScene["showEdges"] = not gPhysicsScene["showEdges"]


# ==============================================================================
# 3. TOPOLOGIA SIATKI: WYSZUKIWANIE SĄSIEDNICH TRÓJKĄTÓW (findTriNeighbors)
# Każdy trójkąt `i` posiada 3 skierowane półkrawędzie (half-edges) o indeksach
# `3*i + 0`, `3*i + 1`, `3*i + 2`. Sortowanie półkrawędzi po parze wierzchołków
# (min, max) w czasie O(M log M) ustawia bliźniacze krawędzie sąsiadów obok siebie.
# ==============================================================================

def findTriNeighbors(triIds):
    edges = []
    numTris = len(triIds) // 3

    for i in range(numTris):
        for j in range(3):
            id0 = int(triIds[3 * i + j])
            id1 = int(triIds[3 * i + (j + 1) % 3])
            edges.append((min(id0, id1), max(id0, id1), 3 * i + j))

    # Sortowanie półkrawędzi tak, aby wspólne krawędzie znalazły się obok siebie
    edges.sort(key=lambda e: (e[0], e[1]))

    neighbors = np.full(3 * numTris, -1, dtype=np.int32)  # -1 oznacza otwartą krawędź brzegową

    nr = 0
    numEdges = len(edges)
    while nr < numEdges - 1:
        e0 = edges[nr]
        e1 = edges[nr + 1]
        if e0[0] == e1[0] and e0[1] == e1[1]:
            neighbors[e0[2]] = e1[2]
            neighbors[e1[2]] = e0[2]
            nr += 2
        else:
            # Krawędź brzegowa (brak pary): przesuwamy wskaźnik dokładnie o 1 element,
            # aby nie pominąć kolejnej krawędzi wewnętrznej
            nr += 1

    return neighbors


# ==============================================================================
# 4. JĄDRA OBLICZENIOWE SOLVERA XPBD (Kompilowane JIT przez Numba @njit)
# ==============================================================================

@njit
def _preSolve_kernel(
    numParticles, pos, prevPos, vel, invMass, normals, dt, gravity,
    windEnabled, windTime, sphereEnabled, sphereGrabbed, spherePos, sphereVel, sphereRadius
):
    """
    Krok 1 każdego podkroku (preSolve):
    1) Fizyka kuli 3D (grawitacja, odbicie od podłogi y = R, tarcie toczne).
    2) Dla każdej swobodnej cząstki tkaniny (invMass[i] > 0):
       - Całkowanie symplektycznym schematem Eulera (grawitacja + wiatr),
       - Kolizja z podłogą (y < 0) oraz wypchnięcie cząstki na powierzchnię kuli 3D.
    """
    if sphereEnabled and not sphereGrabbed:
        sphereVel[0] += gravity[0] * dt
        sphereVel[1] += gravity[1] * dt
        sphereVel[2] += gravity[2] * dt

        spherePos[0] += sphereVel[0] * dt
        spherePos[1] += sphereVel[1] * dt
        spherePos[2] += sphereVel[2] * dt

        if spherePos[1] < sphereRadius:
            spherePos[1] = sphereRadius
            if sphereVel[1] < 0.0:
                sphereVel[1] = -sphereVel[1] * 0.45
            sphereVel[0] *= 0.992
            sphereVel[2] *= 0.992

        for axis in (0, 2):
            if spherePos[axis] > 0.9:
                spherePos[axis] = 0.9
                sphereVel[axis] = -sphereVel[axis] * 0.5
            elif spherePos[axis] < -0.9:
                spherePos[axis] = -0.9
                sphereVel[axis] = -sphereVel[axis] * 0.5

    wx = 1.5 * math.sin(windTime * 2.5)
    wz = 3.5 * (0.6 + 0.4 * math.cos(windTime * 1.8))

    sx, sy, sz = spherePos[0], spherePos[1], spherePos[2]
    r_cloth = sphereRadius + 0.006

    for i in range(numParticles):
        if invMass[i] == 0.0:
            continue

        # A) Przyspieszenie grawitacyjne: vel += g * dt
        vecAdd(vel, i, gravity, 0, dt)

        # B) DODATKOWA FUNKCJONALNOŚĆ — WIATR:
        if windEnabled:
            nx, nz = normals[3 * i], normals[3 * i + 2]
            pressure = abs(nx * wx + nz * wz) + 0.35
            vel[3 * i] += wx * pressure * dt
            vel[3 * i + 2] += wz * pressure * dt

        # C) Zapisanie poprzedniej pozycji i predykcja nowej pozycji (symplektyczny Euler)
        vecCopy(prevPos, i, pos, i)
        vecAdd(pos, i, vel, i, dt)

        # D) Kolizja cząstki z podłogą (y < 0)
        if pos[3 * i + 1] < 0.0:
            vecCopy(pos, i, prevPos, i)
            pos[3 * i + 1] = 0.0

        # E) DODATKOWA FUNKCJONALNOŚĆ — KOLIZJA Z KULĄ 3D:
        if sphereEnabled:
            dx = pos[3 * i] - sx
            dy = pos[3 * i + 1] - sy
            dz = pos[3 * i + 2] - sz
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < r_cloth * r_cloth and dist2 > 1e-12:
                dist = math.sqrt(dist2)
                inv_dist = 1.0 / dist
                nx_s = dx * inv_dist
                ny_s = dy * inv_dist
                nz_s = dz * inv_dist

                pos[3 * i] = sx + nx_s * r_cloth
                pos[3 * i + 1] = sy + ny_s * r_cloth
                pos[3 * i + 2] = sz + nz_s * r_cloth

                if not sphereGrabbed:
                    vn = sphereVel[0] * nx_s + sphereVel[1] * ny_s + sphereVel[2] * nz_s
                    if vn > 0.0:
                        sphereVel[0] -= 0.0015 * vn * nx_s
                        sphereVel[1] -= 0.0015 * vn * ny_s
                        sphereVel[2] -= 0.0015 * vn * nz_s


@njit
def _solveStretching_kernel(pos, invMass, stretchingIds, stretchingLengths, numStretch, compliance, dt, grads):
    """
    Krok 2a: Rozwiązywanie więzów rozciągania (Stretching) metodą Gaussa-Seidela w XPBD.
    """
    alpha = compliance / dt / dt
    for i in range(numStretch):
        id0 = stretchingIds[2 * i]
        id1 = stretchingIds[2 * i + 1]
        w0 = invMass[id0]
        w1 = invMass[id1]
        w = w0 + w1
        if w == 0.0:
            continue

        vecSetDiff(grads, 0, pos, id0, pos, id1)
        length = math.sqrt(vecLengthSquared(grads, 0))
        if length == 0.0:
            continue
        vecScale(grads, 0, 1.0 / length)
        restLen = stretchingLengths[i]
        C = length - restLen
        s = -C / (w + alpha)
        vecAdd(pos, id0, grads, 0, s * w0)
        vecAdd(pos, id1, grads, 0, -s * w1)


@njit
def _solveBending_kernel(pos, invMass, bendingIds, bendingLengths, numBend, compliance, dt, grads):
    """
    Krok 2b: Szybkie izometryczne więzy zginania (Bending) pomiędzy przeciwległymi
    wierzchołkami (id2, id3) sąsiednich trójkątów. Jeśli więz został przecięty przez
    rozszczepienie wierzchołka (bendingLengths[i] < 0), jest pomijany.
    """
    alpha = compliance / dt / dt
    for i in range(numBend):
        restLen = bendingLengths[i]
        if restLen < 0.0:
            continue

        id0 = bendingIds[4 * i + 2]
        id1 = bendingIds[4 * i + 3]
        w0 = invMass[id0]
        w1 = invMass[id1]
        w = w0 + w1
        if w == 0.0:
            continue

        vecSetDiff(grads, 0, pos, id0, pos, id1)
        length = math.sqrt(vecLengthSquared(grads, 0))
        if length == 0.0:
            continue
        vecScale(grads, 0, 1.0 / length)
        C = length - restLen
        s = -C / (w + alpha)
        vecAdd(pos, id0, grads, 0, s * w0)
        vecAdd(pos, id1, grads, 0, -s * w1)


@njit
def _postSolve_kernel(numParticles, pos, prevPos, vel, invMass, dt):
    """Krok 3 każdego podkroku (postSolve): aktualizacja prędkości vel = (pos - prevPos) / dt."""
    inv_dt = 1.0 / dt
    for i in range(numParticles):
        if invMass[i] == 0.0:
            continue
        vecSetDiff(vel, i, pos, i, prevPos, i, inv_dt)


@njit
def _find_overstretched_edges_kernel(
    pos, restPos, stretchingIds, stretchingLengths, numStretch, threshold, outEdges, outRatios
):
    """
    Szybkie wyszukiwanie listy top-K najbardziej napiętych krawędzi siatki przekraczających
    próg rozciągnięcia (length / restLen > threshold), z pominięciem strefy przy dwóch
    górnych punktach mocowania (restPos_y >= 1.04). Zwracanie listy kandydatów zapobiega
    blokowaniu rozrywania przez pojedynczą narożną krawędź (1 trójkąt) na brzegu szczeliny.
    """
    max_k = len(outEdges)
    count = 0
    for i in range(numStretch):
        id0 = stretchingIds[2 * i]
        id1 = stretchingIds[2 * i + 1]
        if restPos[3 * id0 + 1] >= 1.04 or restPos[3 * id1 + 1] >= 1.04:
            continue
        length = math.sqrt(vecDistSquared(pos, id0, pos, id1))
        ratio = length / stretchingLengths[i]
        if ratio > threshold:
            if count < max_k:
                pos_idx = count
                count += 1
            elif ratio > outRatios[max_k - 1]:
                pos_idx = max_k - 1
            else:
                continue
            while pos_idx > 0 and ratio > outRatios[pos_idx - 1]:
                outEdges[pos_idx] = outEdges[pos_idx - 1]
                outRatios[pos_idx] = outRatios[pos_idx - 1]
                pos_idx -= 1
            outEdges[pos_idx] = i
            outRatios[pos_idx] = ratio
    return count


@njit
def _computeVertexNormals(numParticles, pos, triIds, normals):
    """Oblicza znormalizowane wektory normalne wierzchołków (do oświetlenia Phonga i siły wiatru)."""
    for v in range(numParticles):
        normals[3 * v] = 0.0
        normals[3 * v + 1] = 0.0
        normals[3 * v + 2] = 0.0

    numTris = len(triIds) // 3
    for i in range(numTris):
        i0, i1, i2 = triIds[3 * i], triIds[3 * i + 1], triIds[3 * i + 2]
        p0x, p0y, p0z = pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]
        e0x, e0y, e0z = pos[3 * i1] - p0x, pos[3 * i1 + 1] - p0y, pos[3 * i1 + 2] - p0z
        e1x, e1y, e1z = pos[3 * i2] - p0x, pos[3 * i2 + 1] - p0y, pos[3 * i2 + 2] - p0z
        nx = e0y * e1z - e0z * e1y
        ny = e0z * e1x - e0x * e1z
        nz = e0x * e1y - e0y * e1x
        for v in (i0, i1, i2):
            normals[3 * v] += nx
            normals[3 * v + 1] += ny
            normals[3 * v + 2] += nz

    for v in range(numParticles):
        nx, ny, nz = normals[3 * v], normals[3 * v + 1], normals[3 * v + 2]
        l = math.sqrt(nx * nx + ny * ny + nz * nz)
        if l > 1e-12:
            inv_l = 1.0 / l
            normals[3 * v] *= inv_l
            normals[3 * v + 1] *= inv_l
            normals[3 * v + 2] *= inv_l


# ==============================================================================
# 5. KLASA TKANINY (class Cloth z Dynamic Vertex Splitting)
# ==============================================================================

class Cloth:
    def __init__(self, mesh, bendingCompliance=1.0):
        # Wstępna alokacja buforów z zapasem na nowe wierzchołki i krawędzie powstające podczas rozrywania
        # (dzięki temu Numba nie musi alokować pamięci w trakcie symulacji i zachowuje >200 FPS)
        n0 = len(mesh["vertices"]) // 3
        self.maxVerts = n0 * 3
        self.numParticles = n0

        self.pos = np.zeros(3 * self.maxVerts, dtype=np.float32)
        self.prevPos = np.zeros(3 * self.maxVerts, dtype=np.float32)
        self.restPos = np.zeros(3 * self.maxVerts, dtype=np.float32)
        self.vel = np.zeros(3 * self.maxVerts, dtype=np.float32)
        self.invMass = np.zeros(self.maxVerts, dtype=np.float32)
        self.normals = np.zeros(3 * self.maxVerts, dtype=np.float32)

        self.pos[:3 * n0] = mesh["vertices"]
        self.prevPos[:3 * n0] = mesh["vertices"]
        self.restPos[:3 * n0] = mesh["vertices"]

        self.faceTriIds = np.array(mesh["faceTriIds"], dtype=np.int32)
        self.numTris = len(self.faceTriIds) // 3

        neighbors = findTriNeighbors(self.faceTriIds)

        self.maxEdges = 25000
        self.maxBend = 15000
        self.stretchingIds = np.zeros(2 * self.maxEdges, dtype=np.int32)
        self.stretchingLengths = np.zeros(self.maxEdges, dtype=np.float32)
        self.bendingIds = np.zeros(4 * self.maxBend, dtype=np.int32)
        self.bendingLengths = np.zeros(self.maxBend, dtype=np.float32)
        self.bendTris = np.zeros(2 * self.maxBend, dtype=np.int32)

        # Lokalne listy sąsiedztwa potrzebne do rozszczepiania wierzchołka w czasie O(walencja) = O(6)
        self.vert_tris = [set() for _ in range(self.maxVerts)]
        self.vert_edges = [set() for _ in range(self.maxVerts)]
        self.vert_bends = [set() for _ in range(self.maxVerts)]
        self.edge_tris = []

        edge_map = {}
        e_count = 0
        b_count = 0

        for i in range(self.numTris):
            for k in range(3):
                self.vert_tris[int(self.faceTriIds[3 * i + k])].add(i)
            for j in range(3):
                id0 = int(self.faceTriIds[3 * i + j])
                id1 = int(self.faceTriIds[3 * i + (j + 1) % 3])
                key = (min(id0, id1), max(id0, id1))
                n = int(neighbors[3 * i + j])

                # Każda unikalna krawędź wchodzi do więzów rozciągania dokładnie raz
                if key not in edge_map:
                    edge_map[key] = e_count
                    self.stretchingIds[2 * e_count] = id0
                    self.stretchingIds[2 * e_count + 1] = id1
                    ni = (n // 3) if n >= 0 else -1
                    self.edge_tris.append([i, ni])
                    self.vert_edges[id0].add(e_count)
                    self.vert_edges[id1].add(e_count)
                    e_count += 1

                # Każda wspólna krawędź wewnętrzna dostaje dokładnie jeden więz zginania (bez duplikatów)
                if n > 3 * i + j:
                    ni = n // 3
                    nj = n % 3
                    id2 = int(self.faceTriIds[3 * i + (j + 2) % 3])
                    id3 = int(self.faceTriIds[3 * ni + (nj + 2) % 3])
                    self.bendingIds[4 * b_count:4 * b_count + 4] = [id0, id1, id2, id3]
                    self.bendTris[2 * b_count] = i
                    self.bendTris[2 * b_count + 1] = ni
                    for vid in (id0, id1, id2, id3):
                        self.vert_bends[vid].add(b_count)
                    b_count += 1

        self.numStretch = e_count
        self.numBend = b_count

        self.stretchingCompliance = 0.0
        self.bendingCompliance = float(bendingCompliance)

        self.temp = np.zeros(4 * 3, dtype=np.float32)
        self.grads = np.zeros(4 * 3, dtype=np.float32)
        self.tearCandEdges = np.zeros(16, dtype=np.int32)
        self.tearCandRatios = np.zeros(16, dtype=np.float32)

        # Dane chwytu myszką (zarówno pojedynczego wierzchołka, jak i małego obszaru palców przy rozrywaniu)
        self.grabIds = []
        self.grabInvMasses = []
        self.grabOffsets = []
        self.pinnedSet = set()

        self.initPhysics(self.faceTriIds)
        self.updateMeshes()

    def initPhysics(self, triIds):
        """Oblicza odwrotne masy cząstek z pola powierzchni trójkątów i przypina dwa górne narożniki."""
        self.invMass.fill(0.0)
        self.pinnedSet.clear()

        numTris = len(triIds) // 3
        e0 = np.zeros(3, dtype=np.float32)
        e1 = np.zeros(3, dtype=np.float32)
        c = np.zeros(3, dtype=np.float32)

        for i in range(numTris):
            id0 = int(triIds[3 * i])
            id1 = int(triIds[3 * i + 1])
            id2 = int(triIds[3 * i + 2])
            vecSetDiff(e0, 0, self.pos, id1, self.pos, id0)
            vecSetDiff(e1, 0, self.pos, id2, self.pos, id0)
            vecSetCross(c, 0, e0, 0, e1, 0)
            A = 0.5 * math.sqrt(vecLengthSquared(c, 0))
            pInvMass = (1.0 / A / 3.0) if A > 0.0 else 0.0
            self.invMass[id0] += pInvMass
            self.invMass[id1] += pInvMass
            self.invMass[id2] += pInvMass

        for i in range(self.numStretch):
            id0 = int(self.stretchingIds[2 * i])
            id1 = int(self.stretchingIds[2 * i + 1])
            self.stretchingLengths[i] = math.sqrt(vecDistSquared(self.pos, id0, self.pos, id1))

        for i in range(self.numBend):
            id0 = int(self.bendingIds[4 * i + 2])
            id1 = int(self.bendingIds[4 * i + 3])
            self.bendingLengths[i] = math.sqrt(vecDistSquared(self.pos, id0, self.pos, id1))

        # Przypięcie dwóch górnych narożników tkaniny (invMass = 0)
        minX = float("inf")
        maxX = float("-inf")
        maxY = float("-inf")

        for i in range(self.numParticles):
            minX = min(minX, float(self.pos[3 * i]))
            maxX = max(maxX, float(self.pos[3 * i]))
            maxY = max(maxY, float(self.pos[3 * i + 1]))

        eps = 0.0001
        for i in range(self.numParticles):
            x = self.pos[3 * i]
            y = self.pos[3 * i + 1]
            if (y > maxY - eps) and (x < minX + eps or x > maxX - eps):
                self.invMass[i] = 0.0
                self.pinnedSet.add(i)

    def splitVertex(self, v, nx, ny):
        """
        Algorytm Dynamic Vertex Splitting (bezpieczne rozrywanie siatki bez usuwania trójkątów):
        Rozszczepia wierzchołek `v` wzdłuż płaszczyzny cięcia o normalnej (nx, ny) w restPos:
        1) Dzieli wachlarz trójkątów wokół `v` na dwie spójne strony: T_A (zostaje z `v`) i T_B (dostaje `v_new`).
        2) Tworzy duplikat cząstki `v_new` z zachowaniem sumy mas (m_A + m_B = m_v).
        3) Przepisuje trójkąty z T_B na `v_new`, aktualizuje lub duplikuje krawędzie szwu (Golden Invariant:
           każdy trójkąt zachowuje wszystkie 3 krawędzie, więc nigdy nie powstaje gigantyczny rozciągnięty trójkąt!).
        4) Wyłącza więzy zginania przekraczające linię rozcięcia (zero niewidzialnego wiszenia w powietrzu).
        """
        if self.numParticles >= self.maxVerts - 1 or v in self.pinnedSet:
            return False

        tris = list(self.vert_tris[v])
        if len(tris) < 2:
            return False

        # Budujemy graf sąsiedztwa trójkątów wokół `v` przez wspólne krawędzie wewnętrzne
        full_adj = {t: [] for t in tris}
        for e in self.vert_edges[v]:
            t0, t1 = self.edge_tris[e]
            if t0 in full_adj and t1 in full_adj:
                full_adj[t0].append(t1)
                full_adj[t1].append(t0)

        # Krok 0 (Pinch vertex): jeśli wachlarz wokół `v` składa się już z 2+ rozłącznych
        # płatów połączonych tylko jednym wspólnym punktem (częste na brzegu starego rozdarcia),
        # natychmiast je rozpinamy bez blokowania dalszego rozrywania!
        fan0 = {tris[0]}
        stack = [tris[0]]
        while stack:
            cur = stack.pop()
            for nxt in full_adj[cur]:
                if nxt not in fan0:
                    fan0.add(nxt)
                    stack.append(nxt)

        if len(fan0) < len(tris):
            T_B = fan0
            T_A = set(tris) - fan0
        else:
            rvx, rvy = float(self.restPos[3 * v]), float(self.restPos[3 * v + 1])
            T_A = set()
            T_B = set()
            dots = {}

            for t in tris:
                i0, i1, i2 = int(self.faceTriIds[3 * t]), int(self.faceTriIds[3 * t + 1]), int(self.faceTriIds[3 * t + 2])
                cx = (self.restPos[3 * i0] + self.restPos[3 * i1] + self.restPos[3 * i2]) / 3.0 - rvx
                cy = (self.restPos[3 * i0 + 1] + self.restPos[3 * i1 + 1] + self.restPos[3 * i2 + 1]) / 3.0 - rvy
                d = float(cx * nx + cy * ny)
                dots[t] = d
                if d >= 0.0:
                    T_A.add(t)
                else:
                    T_B.add(t)

            # Jeśli wierzchołek leży na brzegu szczeliny i płaszczyzna ominęła pół-wachlarz,
            # dzielimy trójkąty w połowie wzdłuż rzutu `dots`, aby szczelina mogła iść dalej.
            if not T_A or not T_B:
                ordered = sorted(tris, key=lambda t: dots[t])
                half = len(ordered) // 2
                T_B = set(ordered[:half])
                T_A = set(ordered[half:])

            # Zachowujemy spójność topologiczną wachlarza T_B wokół wierzchołka `v` (2-manifold)
            seed_B = min(T_B, key=lambda t: dots[t])
            comp_B = {seed_B}
            stack = [seed_B]
            while stack:
                cur = stack.pop()
                for nxt in full_adj[cur]:
                    if nxt in T_B and nxt not in comp_B:
                        comp_B.add(nxt)
                        stack.append(nxt)
            if len(comp_B) < len(T_B):
                T_A = T_A | (T_B - comp_B)
                T_B = comp_B

            # Sprawdzamy również spójność strony T_A (dla otwartego pół-wachlarza na brzegu
            # cięcie mogło trafić w środek łańcucha — wtedy tniemy od końca łańcucha w połowie).
            seed_A = max(T_A, key=lambda t: dots[t])
            comp_A = {seed_A}
            stack = [seed_A]
            while stack:
                cur = stack.pop()
                for nxt in full_adj[cur]:
                    if nxt in T_A and nxt not in comp_A:
                        comp_A.add(nxt)
                        stack.append(nxt)
            if len(comp_A) < len(T_A):
                endpoints = [t for t in tris if len(full_adj[t]) <= 1]
                if not endpoints:
                    return False
                start_t = min(endpoints, key=lambda t: dots[t])
                chain = [start_t]
                prev_t = -1
                cur_t = start_t
                while len(chain) < len(tris):
                    nxts = [nt for nt in full_adj[cur_t] if nt != prev_t]
                    if not nxts:
                        break
                    prev_t, cur_t = cur_t, nxts[0]
                    chain.append(cur_t)
                if len(chain) < 2:
                    return False
                half = len(chain) // 2
                T_B = set(chain[:half])
                T_A = set(chain[half:])
                if not T_A or not T_B:
                    return False

        # Alokacja nowego wierzchołka v_new i skopiowanie jego stanu kinematycznego
        v_new = self.numParticles
        self.numParticles += 1
        self.pos[3 * v_new:3 * v_new + 3] = self.pos[3 * v:3 * v + 3]
        self.prevPos[3 * v_new:3 * v_new + 3] = self.prevPos[3 * v:3 * v + 3]
        self.restPos[3 * v_new:3 * v_new + 3] = self.restPos[3 * v:3 * v + 3]
        self.vel[3 * v_new:3 * v_new + 3] = self.vel[3 * v:3 * v + 3]

        # Podział masy cząstki w stosunku proporcjonalnym do liczby trójkątów (zachowanie masy całkowitej)
        frac_A = len(T_A) / len(tris)
        if v in self.grabIds:
            idx_g = self.grabIds.index(v)
            orig_w = self.grabInvMasses[idx_g]
            self.grabInvMasses[idx_g] = orig_w / frac_A
            self.invMass[v_new] = orig_w / (1.0 - frac_A)
        else:
            orig_w = float(self.invMass[v])
            self.invMass[v] = orig_w / frac_A
            self.invMass[v_new] = orig_w / (1.0 - frac_A)

        # Przepięcie trójkątów strony T_B z wierzchołka v na v_new
        for t in T_B:
            for k in range(3):
                if self.faceTriIds[3 * t + k] == v:
                    self.faceTriIds[3 * t + k] = v_new
            self.vert_tris[v].remove(t)
            self.vert_tris[v_new].add(t)

        # Aktualizacja lub duplikacja krawędzi rozciągania (gwarantuje kompletność każdego trójkąta)
        for e in list(self.vert_edges[v]):
            a, b = int(self.stretchingIds[2 * e]), int(self.stretchingIds[2 * e + 1])
            nb = b if a == v else a
            t0, t1 = self.edge_tris[e]
            t0_A, t0_B = (t0 in T_A), (t0 in T_B)
            t1_A, t1_B = (t1 >= 0 and t1 in T_A), (t1 >= 0 and t1 in T_B)
            has_A, has_B = (t0_A or t1_A), (t0_B or t1_B)

            if has_A and not has_B:
                continue
            elif has_B and not has_A:
                if a == v:
                    self.stretchingIds[2 * e] = v_new
                else:
                    self.stretchingIds[2 * e + 1] = v_new
                self.vert_edges[v].remove(e)
                self.vert_edges[v_new].add(e)
            elif has_A and has_B:
                # Krawędź szwu pęknięcia: strona T_A zachowuje krawędź `e`, strona T_B dostaje duplikat `e_new`
                if self.numStretch >= self.maxEdges - 1:
                    continue
                t_A = t0 if t0_A else t1
                t_B = t0 if t0_B else t1
                self.edge_tris[e] = [t_A, -1]
                e_new = self.numStretch
                self.numStretch += 1
                self.stretchingIds[2 * e_new] = v_new
                self.stretchingIds[2 * e_new + 1] = nb
                self.stretchingLengths[e_new] = self.stretchingLengths[e]
                self.edge_tris.append([t_B, -1])
                self.vert_edges[v_new].add(e_new)
                self.vert_edges[nb].add(e_new)

        # Aktualizacja lub usunięcie więzów zginania przy rozciętej krawędzi
        for b_idx in list(self.vert_bends[v]):
            if self.bendingLengths[b_idx] < 0.0:
                continue
            t1, t2 = int(self.bendTris[2 * b_idx]), int(self.bendTris[2 * b_idx + 1])
            id0, id1, id2, id3 = [int(x) for x in self.bendingIds[4 * b_idx:4 * b_idx + 4]]

            if id0 == v or id1 == v:
                t1_A, t2_A = (t1 in T_A), (t2 in T_A)
                if t1_A and t2_A:
                    continue
                elif (not t1_A) and (not t2_A):
                    for k in range(4):
                        if self.bendingIds[4 * b_idx + k] == v:
                            self.bendingIds[4 * b_idx + k] = v_new
                    self.vert_bends[v].remove(b_idx)
                    self.vert_bends[v_new].add(b_idx)
                else:
                    # Krawędź zginania przecina szczelinę — wyłączamy ją, aby połówki nie wisiały w powietrzu!
                    self.bendingLengths[b_idx] = -1.0
                    self.vert_bends[v].discard(b_idx)
            else:
                if id2 == v and (t1 in T_B):
                    self.bendingIds[4 * b_idx + 2] = v_new
                    self.vert_bends[v].remove(b_idx)
                    self.vert_bends[v_new].add(b_idx)
                elif id3 == v and (t2 in T_B):
                    self.bendingIds[4 * b_idx + 3] = v_new
                    self.vert_bends[v].remove(b_idx)
                    self.vert_bends[v_new].add(b_idx)

        return True

    def tryTearStep(self, max_splits=2):
        """
        Sprawdza top-K najbardziej napiętych krawędzi po relaksacji podkroków i rozszczepia
        ich wierzchołki (do `max_splits` na wywołanie), pomijając zablokowane rogi 1-trójkątowe.
        """
        n_cand = _find_overstretched_edges_kernel(
            self.pos, self.restPos, self.stretchingIds, self.stretchingLengths,
            self.numStretch, gPhysicsScene["tearThreshold"],
            self.tearCandEdges, self.tearCandRatios
        )
        if n_cand == 0:
            return False

        splits = 0
        for k in range(n_cand):
            e = int(self.tearCandEdges[k])
            if e >= self.numStretch:
                continue
            id0 = int(self.stretchingIds[2 * e])
            id1 = int(self.stretchingIds[2 * e + 1])
            nx = float(self.restPos[3 * id1] - self.restPos[3 * id0])
            ny = float(self.restPos[3 * id1 + 1] - self.restPos[3 * id0 + 1])
            # Preferujemy rozszczepienie wierzchołka o większej liczbie trójkątów (lub spoza chwytu)
            if id0 in self.grabIds and id1 not in self.grabIds:
                order = ((id1, -nx, -ny), (id0, nx, ny))
            elif id1 in self.grabIds and id0 not in self.grabIds:
                order = ((id0, nx, ny), (id1, -nx, -ny))
            elif len(self.vert_tris[id0]) >= len(self.vert_tris[id1]):
                order = ((id0, nx, ny), (id1, -nx, -ny))
            else:
                order = ((id1, -nx, -ny), (id0, nx, ny))

            for vid, cnx, cny in order:
                if self.splitVertex(vid, cnx, cny):
                    splits += 1
                    break
            if splits >= max_splits:
                break
        return splits > 0

    def preSolve(self, dt, gravity):
        _preSolve_kernel(
            self.numParticles,
            self.pos,
            self.prevPos,
            self.vel,
            self.invMass,
            self.normals,
            dt,
            gravity,
            gPhysicsScene["windEnabled"],
            time.perf_counter(),
            gPhysicsScene["sphereEnabled"],
            gPhysicsScene["sphereGrabbed"],
            gPhysicsScene["spherePos"],
            gPhysicsScene["sphereVel"],
            gPhysicsScene["sphereRadius"],
        )

    def solve(self, dt):
        self.solveStretching(self.stretchingCompliance, dt)
        self.solveBending(self.bendingCompliance, dt)

    def postSolve(self, dt):
        _postSolve_kernel(self.numParticles, self.pos, self.prevPos, self.vel, self.invMass, dt)

    def solveStretching(self, compliance, dt):
        _solveStretching_kernel(
            self.pos, self.invMass, self.stretchingIds, self.stretchingLengths,
            self.numStretch, float(compliance), dt, self.grads
        )

    def solveBending(self, compliance, dt):
        _solveBending_kernel(
            self.pos, self.invMass, self.bendingIds, self.bendingLengths,
            self.numBend, float(compliance), dt, self.grads
        )

    def updateMeshes(self):
        _computeVertexNormals(self.numParticles, self.pos, self.faceTriIds, self.normals)

    def endFrame(self):
        self.updateMeshes()

    def startGrab(self, pos, radius=0.0):
        """
        Chwyta wierzchołki tkaniny w pobliżu trafienia promieniem myszy.
        W trybie Tear cloth chwyta obszar o promieniu ~2.5 cm (jak palce dłoni), aby
        delikatne pociągnięcie nie wyrywało pojedynczego atomu siatki.
        """
        p = np.array([pos[0], pos[1], pos[2]], dtype=np.float32)
        self.grabIds.clear()
        self.grabInvMasses.clear()
        self.grabOffsets.clear()

        minD2 = float("inf")
        closest_id = -1
        for i in range(self.numParticles):
            d2 = vecDistSquared(p, 0, self.pos, i)
            if d2 < minD2:
                minD2 = d2
                closest_id = i

        if closest_id < 0:
            return

        if radius > 0.0:
            r2 = radius * radius
            for i in range(self.numParticles):
                if i in self.pinnedSet:
                    continue
                if vecDistSquared(self.pos, closest_id, self.pos, i) <= r2:
                    self.grabIds.append(i)
                    self.grabInvMasses.append(float(self.invMass[i]))
                    offset = self.pos[3 * i:3 * i + 3] - p
                    self.grabOffsets.append(offset.copy())
                    self.invMass[i] = 0.0
        elif closest_id not in self.pinnedSet:
            self.grabIds.append(closest_id)
            self.grabInvMasses.append(float(self.invMass[closest_id]))
            self.grabOffsets.append(np.zeros(3, dtype=np.float32))
            self.invMass[closest_id] = 0.0
            vecCopy(self.pos, closest_id, p, 0)

    def moveGrabbed(self, pos, vel):
        """Przesuwa chwycone wierzchołki zgodnie z ruchem kursora myszy (nie poniżej podłogi y = 0)."""
        p = np.array([pos[0], pos[1], pos[2]], dtype=np.float32)
        for idx, vid in enumerate(self.grabIds):
            target = p + self.grabOffsets[idx]
            if target[1] < 0.002:
                target[1] = 0.002
            self.pos[3 * vid:3 * vid + 3] = target

    def endGrab(self, pos, vel):
        """Puszcza chwycone wierzchołki: przywraca ich masy i nadaje prędkość ruchu myszy."""
        v = np.array([vel[0], vel[1], vel[2]], dtype=np.float32)
        for idx, vid in enumerate(self.grabIds):
            self.invMass[vid] = self.grabInvMasses[idx]
            vecCopy(self.vel, vid, v, 0)
        self.grabIds.clear()
        self.grabInvMasses.clear()
        self.grabOffsets.clear()


# ==============================================================================
# 6. CHWYTANIE I RZUCANIE MYSZKĄ ZARÓWNO TKANINY, JAK I KULI 3D (Grabber)
# ==============================================================================

@njit
def _raycast_mesh(ray_orig, ray_dir, pos, triIds):
    """Test przecięcia promienia z siatką trójkątów tkaniny (algorytm Möllera-Trumbore'a)."""
    numTris = len(triIds) // 3
    best_t = 1e30
    hit = False
    ox, oy, oz = ray_orig[0], ray_orig[1], ray_orig[2]
    dx, dy, dz = ray_dir[0], ray_dir[1], ray_dir[2]

    for i in range(numTris):
        i0, i1, i2 = triIds[3 * i], triIds[3 * i + 1], triIds[3 * i + 2]
        v0x, v0y, v0z = pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]
        e1x, e1y, e1z = pos[3 * i1] - v0x, pos[3 * i1 + 1] - v0y, pos[3 * i1 + 2] - v0z
        e2x, e2y, e2z = pos[3 * i2] - v0x, pos[3 * i2 + 1] - v0y, pos[3 * i2 + 2] - v0z
        px = dy * e2z - dz * e2y
        py = dz * e2x - dx * e2z
        pz = dx * e2y - dy * e2x
        det = e1x * px + e1y * py + e1z * pz
        if abs(det) < 1e-8:
            continue
        inv_det = 1.0 / det
        tx, ty, tz = ox - v0x, oy - v0y, oz - v0z
        u = (tx * px + ty * py + tz * pz) * inv_det
        if u < 0.0 or u > 1.0:
            continue
        qx = ty * e1z - tz * e1y
        qy = tz * e1x - tx * e1z
        qz = tx * e1y - ty * e1x
        v = (dx * qx + dy * qy + dz * qz) * inv_det
        if v < 0.0 or u + v > 1.0:
            continue
        t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
        if 0.001 < t < best_t:
            best_t = t
            hit = True
    return hit, best_t


@njit
def _raycast_sphere(ray_orig, ray_dir, spherePos, sphereRadius):
    """Analityczny test przecięcia promienia myszy z kulą 3D: ||o + t*d - c||^2 = R^2."""
    ocx = ray_orig[0] - spherePos[0]
    ocy = ray_orig[1] - spherePos[1]
    ocz = ray_orig[2] - spherePos[2]
    b_half = ocx * ray_dir[0] + ocy * ray_dir[1] + ocz * ray_dir[2]
    c = ocx * ocx + ocy * ocy + ocz * ocz - sphereRadius * sphereRadius
    disc = b_half * b_half - c
    if disc >= 0.0:
        t = -b_half - math.sqrt(disc)
        if t > 0.01:
            return True, t
    return False, 1e30


class Grabber:
    """
    Obsługuje interakcję myszką w przestrzeni 3D:
    - Kliknięcie na kuli 3D pozwala podnosić kulę, pchać ją w tkaninę i rzucać w nią myszką.
    - Kliknięcie na tkaninie chwyta jej powierzchnię i pozwala ciągnąć lub rozrywać siatkę.
    """
    def __init__(self, camera):
        self.camera = camera
        self.physicsObject = None
        self.grabbedSphere = False
        self.distance = 0.0
        self.prevPos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.time = 0.0
        self.ray_origin = np.zeros(3, dtype=np.float32)
        self.ray_direction = np.zeros(3, dtype=np.float32)

    @property
    def is_active(self):
        return (self.physicsObject is not None) or self.grabbedSphere

    def increaseTime(self, dt):
        self.time += dt

    def updateRaycaster(self, x, y, width, height):
        self.ray_origin, self.ray_direction = self.camera.get_ray(x, y, width, height)

    def start(self, x, y, width, height, on_run_callback):
        self.physicsObject = None
        self.grabbedSphere = False
        self.updateRaycaster(x, y, width, height)

        hit_sph, dist_sph = False, 1e30
        if gPhysicsScene["sphereEnabled"]:
            hit_sph, dist_sph = _raycast_sphere(
                self.ray_origin, self.ray_direction, gPhysicsScene["spherePos"], gPhysicsScene["sphereRadius"]
            )

        hit_cloth, dist_cloth, cloth_obj = False, 1e30, None
        for obj in gPhysicsScene["objects"]:
            h, d = _raycast_mesh(self.ray_origin, self.ray_direction, obj.pos, obj.faceTriIds)
            if h and d < dist_cloth:
                hit_cloth, dist_cloth, cloth_obj = True, d, obj

        if hit_sph and dist_sph <= dist_cloth:
            self.grabbedSphere = True
            gPhysicsScene["sphereGrabbed"] = True
            gPhysicsScene["sphereVel"][:] = 0.0
            self.distance = float(np.linalg.norm(gPhysicsScene["spherePos"] - self.ray_origin))
            self.prevPos[:] = gPhysicsScene["spherePos"]
            self.vel.fill(0.0)
            self.time = 0.0
            if gPhysicsScene["paused"]:
                on_run_callback()
        elif hit_cloth and cloth_obj is not None:
            self.physicsObject = cloth_obj
            self.distance = dist_cloth
            pos = self.ray_origin + self.ray_direction * self.distance
            grab_radius = 0.025 if gPhysicsScene["tearEnabled"] else 0.0
            self.physicsObject.startGrab(pos, radius=grab_radius)
            self.prevPos[:] = pos
            self.vel.fill(0.0)
            self.time = 0.0
            if gPhysicsScene["paused"]:
                on_run_callback()

    def move(self, x, y, width, height):
        if not self.is_active:
            return
        self.updateRaycaster(x, y, width, height)
        pos = self.ray_origin + self.ray_direction * self.distance

        if self.grabbedSphere:
            r = gPhysicsScene["sphereRadius"]
            if pos[1] < r:
                pos[1] = r
            dt_move = max(self.time, 1.0 / 60.0)
            inst_vel = (pos - self.prevPos) / dt_move
            _, _, _, cam_fwd = self.camera.basis()
            upward_throw = max(0.0, float(inst_vel[1])) * 0.65
            self.vel[:] = 0.5 * self.vel + 0.5 * (inst_vel + cam_fwd * upward_throw)
            self.prevPos[:] = pos
            self.time = 0.0
            gPhysicsScene["spherePos"][:] = pos
            gPhysicsScene["sphereVel"][:] = 0.0
        elif self.physicsObject is not None:
            self.vel[:] = pos - self.prevPos
            if self.time > 0.0:
                self.vel /= self.time
            else:
                self.vel.fill(0.0)
            self.prevPos[:] = pos
            self.time = 0.0
            self.physicsObject.moveGrabbed(pos, self.vel)

    def adjustDistance(self, wheel_delta):
        """Pozwala przybliżać lub oddalać chwyconą kulę w głąb sceny za pomocą kółka myszy."""
        if self.grabbedSphere:
            step = -0.08 if wheel_delta > 0 else 0.08
            self.distance = float(np.clip(self.distance + step, 0.25, 3.0))
            pos = self.ray_origin + self.ray_direction * self.distance
            r = gPhysicsScene["sphereRadius"]
            if pos[1] < r:
                pos[1] = r
            gPhysicsScene["spherePos"][:] = pos
            return True
        return False

    def end(self):
        if self.grabbedSphere:
            gPhysicsScene["sphereGrabbed"] = False
            speed = float(np.linalg.norm(self.vel))
            if speed > 8.5:
                self.vel *= 8.5 / speed
            gPhysicsScene["sphereVel"][:] = self.vel
            self.grabbedSphere = False
        elif self.physicsObject is not None:
            self.physicsObject.endGrab(self.prevPos, self.vel)
            self.physicsObject = None


# ==============================================================================
# 7. WCZYTYWANIE SIATKI Z 14-cloth.html I GŁÓWNA PĘTLA SYMULACJI (simulate)
# ==============================================================================

def load_meshes_from_html():
    """Wczytuje tablicę wierzchołków (vertices) i trójkątów (faceTriIds) bezpośrednio z pliku 14-cloth.html."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "14-cloth.html")
    with open(html_path, "r", encoding="utf-8") as f:
        text = f.read()
    verts_str = re.search(r"vertices\s*:\s*\[([^\]]+)\]", text, re.DOTALL).group(1)
    tris_str = re.search(r"faceTriIds\s*:\s*\[([^\]]+)\]", text, re.DOTALL).group(1)
    vertices = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", verts_str)]
    faceTriIds = [int(x) for x in re.findall(r"\d+", tris_str)]
    return [{"name": "cloth", "vertices": vertices, "faceTriIds": faceTriIds}]


meshes = load_meshes_from_html()
gGrabber = None
timeFrames = 0
timeSum = 0.0
msDisplay = "0.000"
fpsDisplay = "0"


def initPhysics():
    mesh = meshes[0]
    gPhysicsScene["objects"].clear()
    body = Cloth(mesh)
    gPhysicsScene["objects"].append(body)
    return mesh


def simulate():
    """
    Główna funkcja kroku fizyki (15 podkroków XPBD na klatkę).
    Jeśli włączone jest rozrywanie (tearEnabled), co 3 podkroki sprawdzamy przeciążone
    krawędzie — rozszczepienie wierzchołka przeplata się z podkrokami relaksacji, dzięki
    czemu pęknięcie propaguje się płynnie i nie blokuje pętli FPS.
    """
    global timeFrames, timeSum, msDisplay, fpsDisplay
    if gPhysicsScene["paused"]:
        return

    startTime = time.perf_counter()

    sdt = gPhysicsScene["dt"] / gPhysicsScene["numSubsteps"]
    tear_on = gPhysicsScene["tearEnabled"]

    for sub in range(gPhysicsScene["numSubsteps"]):
        for obj in gPhysicsScene["objects"]:
            obj.preSolve(sdt, gPhysicsScene["gravity"])
        for obj in gPhysicsScene["objects"]:
            obj.solve(sdt)
        for obj in gPhysicsScene["objects"]:
            obj.postSolve(sdt)
        if tear_on and (sub % 3 == 2):
            for obj in gPhysicsScene["objects"]:
                obj.tryTearStep()

    for obj in gPhysicsScene["objects"]:
        obj.endFrame()

    if gGrabber is not None:
        gGrabber.increaseTime(gPhysicsScene["dt"])

    endTime = time.perf_counter()
    timeSum += (endTime - startTime) * 1000.0
    timeFrames += 1

    if timeFrames > 10:
        timeSum /= timeFrames
        msDisplay = f"{timeSum:.3f}"
        fpsDisplay = f"{1000.0 / timeSum:.0f}" if timeSum > 1e-6 else "0"
        timeFrames = 0
        timeSum = 0.0


# ==============================================================================
# 8. KAMERA 3D I RASTERYZATOR SCENY Z BUFOREM GŁĘBOKOŚCI (Odpowiednik Three.js)
# ==============================================================================

class OrbitCamera:
    """Kamera orbitalna (OrbitControls) obsługująca obrót wokół sceny i rzutowanie promieni myszy."""
    def __init__(self):
        self.fov = 70.0
        self.target = np.array([0.0, 0.6, 0.0], dtype=np.float32)
        offset = np.array([0.0, 1.0, 1.0], dtype=np.float32) - self.target
        self.radius = float(np.linalg.norm(offset))
        self.theta = math.atan2(float(offset[0]), float(offset[2]))
        self.phi = math.acos(float(offset[1]) / self.radius)

    def basis(self):
        sin_p = math.sin(self.phi)
        pos = self.target + np.array(
            [
                self.radius * sin_p * math.sin(self.theta),
                self.radius * math.cos(self.phi),
                self.radius * sin_p * math.cos(self.theta),
            ],
            dtype=np.float32,
        )
        fwd = self.target - pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return pos, right, up, fwd

    def get_ray(self, x, y, width, height):
        ndc_x = (x / max(1, width)) * 2.0 - 1.0
        ndc_y = -(y / max(1, height)) * 2.0 + 1.0
        aspect = width / max(1, height)
        tan_half = math.tan(math.radians(self.fov) * 0.5)
        pos, right, up, fwd = self.basis()
        d = fwd + (ndc_x * aspect * tan_half) * right + (ndc_y * tan_half) * up
        d /= np.linalg.norm(d)
        return pos, d


@njit
def _draw_line(fb, zbuf, x0, y0, z0, x1, y1, z1):
    """Rysuje pojedynczą krawędź 3D w trybie 'Show edges' z uwzględnieniem Z-bufora."""
    h, w, _ = fb.shape
    steps = int(max(abs(x1 - x0), abs(y1 - y0)))
    if steps == 0:
        return
    if steps > 1500:
        steps = 1500
    for k in range(steps + 1):
        t = k / steps
        px = int(x0 + (x1 - x0) * t)
        py = int(y0 + (y1 - y0) * t)
        pz = z0 + (z1 - z0) * t
        if 0 <= px < w and 0 <= py < h and pz <= zbuf[py, px]:
            zbuf[py, px] = pz
            fb[py, px, 0] = 255
            fb[py, px, 1] = 0
            fb[py, px, 2] = 0


@njit
def _render_three_scene(
    fb, zbuf, numParticles, pos, normals, triIds, edgeIds, numStretch,
    showEdges, sphereEnabled, spherePos, sphereRadius,
    cam_pos, cam_r, cam_u, cam_f, fov_rad
):
    """Renderuje podłogę z siatką, kulę 3D oraz siatkę tkaniny z buforem głębokości (Z-buffer)."""
    h, w, _ = fb.shape
    aspect = w / h
    tan_half = math.tan(fov_rad * 0.5)
    fx_s = 0.5 * w / (aspect * tan_half)
    fy_s = 0.5 * h / tan_half
    cx, cy = 0.5 * w, 0.5 * h
    cpx, cpy, cpz = cam_pos[0], cam_pos[1], cam_pos[2]
    rx, ry, rz = cam_r[0], cam_r[1], cam_r[2]
    ux, uy, uz = cam_u[0], cam_u[1], cam_u[2]
    fx, fy, fz = cam_f[0], cam_f[1], cam_f[2]

    spx, spy, spz = spherePos[0], spherePos[1], spherePos[2]
    ocx, ocy, ocz = cpx - spx, cpy - spy, cpz - spz
    oc_dot_oc = ocx * ocx + ocy * ocy + ocz * ocz - sphereRadius * sphereRadius
    l1x, l1y, l1z = 0.4264, 0.6396, 0.6396

    for py in range(h):
        ny = (cy - (py + 0.5)) / fy_s
        bx, by, bz = fx + ny * ux, fy + ny * uy, fz + ny * uz
        for px in range(w):
            zbuf[py, px] = 1e9
            nx = ((px + 0.5) - cx) / fx_s
            rdx, rdy, rdz = bx + nx * rx, by + nx * ry, bz + nx * rz
            inv_r = 1.0 / math.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
            rdx *= inv_r
            rdy *= inv_r
            rdz *= inv_r

            if rdy < -1e-4 and cpy > 0.0:
                t_hit = -cpy / rdy
                hx, hz = cpx + t_hit * rdx, cpz + t_hit * rdz
                dist = t_hit
                if dist < 15.0 and abs(hx) <= 10.0 and abs(hz) <= 10.0:
                    fog = max(0.0, min(1.0, (15.0 - dist) / 15.0))
                    lw = 0.012 * (1.0 + 0.25 * dist)
                    if abs(hx - round(hx)) < lw or abs(hz - round(hz)) < lw:
                        br, bg, bb = 95.0, 102.0, 105.0
                    else:
                        br, bg, bb = 160.0, 173.0, 175.0
                    if sphereEnabled:
                        ds2 = (hx - spx) ** 2 + (hz - spz) ** 2
                        if ds2 < sphereRadius * sphereRadius * 1.3:
                            br *= 0.65
                            bg *= 0.65
                            bb *= 0.65
                    fb[py, px, 0] = int(br * fog)
                    fb[py, px, 1] = int(bg * fog)
                    fb[py, px, 2] = int(bb * fog)
                    zbuf[py, px] = t_hit * (rdx * fx + rdy * fy + rdz * fz)
                else:
                    fb[py, px, 0] = fb[py, px, 1] = fb[py, px, 2] = 0
            else:
                fb[py, px, 0] = fb[py, px, 1] = fb[py, px, 2] = 0

            if sphereEnabled:
                b_half = ocx * rdx + ocy * rdy + ocz * rdz
                disc = b_half * b_half - oc_dot_oc
                if disc >= 0.0:
                    t_sph = -b_half - math.sqrt(disc)
                    if t_sph > 0.02:
                        z_depth = t_sph * (rdx * fx + rdy * fy + rdz * fz)
                        if z_depth < zbuf[py, px]:
                            zbuf[py, px] = z_depth
                            snx = (cpx + t_sph * rdx - spx) / sphereRadius
                            sny = (cpy + t_sph * rdy - spy) / sphereRadius
                            snz = (cpz + t_sph * rdz - spz) / sphereRadius
                            diff = max(0.0, snx * l1x + sny * l1y + snz * l1z)
                            shade = 0.25 + 0.75 * diff
                            fb[py, px, 0] = int(70 * shade)
                            fb[py, px, 1] = int(145 * shade)
                            fb[py, px, 2] = int(235 * shade)

    sx = np.empty(numParticles, dtype=np.float32)
    sy = np.empty(numParticles, dtype=np.float32)
    sz = np.empty(numParticles, dtype=np.float32)

    for i in range(numParticles):
        vx, vy, vz = pos[3 * i] - cpx, pos[3 * i + 1] - cpy, pos[3 * i + 2] - cpz
        zc = vx * fx + vy * fy + vz * fz
        sz[i] = zc
        if zc > 0.01:
            sx[i] = cx + (vx * rx + vy * ry + vz * rz) / zc * fx_s
            sy[i] = cy - (vx * ux + vy * uy + vz * uz) / zc * fy_s
        else:
            sx[i] = sy[i] = -1e6

    if showEdges:
        for i in range(numStretch):
            i0, i1 = edgeIds[2 * i], edgeIds[2 * i + 1]
            if sz[i0] > 0.02 and sz[i1] > 0.02:
                _draw_line(fb, zbuf, sx[i0], sy[i0], sz[i0], sx[i1], sy[i1], sz[i1])
    else:
        for i in range(len(triIds) // 3):
            i0, i1, i2 = triIds[3 * i], triIds[3 * i + 1], triIds[3 * i + 2]
            z0, z1, z2 = sz[i0], sz[i1], sz[i2]
            if z0 <= 0.02 or z1 <= 0.02 or z2 <= 0.02:
                continue
            x0, y0, x1, y1, x2, y2 = sx[i0], sy[i0], sx[i1], sy[i1], sx[i2], sy[i2]
            min_x = max(0, int(min(x0, x1, x2)))
            max_x = min(w - 1, int(max(x0, x1, x2)) + 1)
            min_y = max(0, int(min(y0, y1, y2)))
            max_y = min(h - 1, int(max(y0, y1, y2)) + 1)
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-5:
                continue
            inv_d = 1.0 / denom
            nx = (normals[3 * i0] + normals[3 * i1] + normals[3 * i2]) / 3.0
            ny = (normals[3 * i0 + 1] + normals[3 * i1 + 1] + normals[3 * i2 + 1]) / 3.0
            nz = (normals[3 * i0 + 2] + normals[3 * i1 + 2] + normals[3 * i2 + 2]) / 3.0
            ndotl = abs(nx * l1x + ny * l1y + nz * l1z)
            c_red = min(255, int(255.0 * (0.32 + 0.68 * ndotl)))
            for py in range(min_y, max_y + 1):
                fy_p = py + 0.5
                for px in range(min_x, max_x + 1):
                    fx_p = px + 0.5
                    w0 = ((y1 - y2) * (fx_p - x2) + (x2 - x1) * (fy_p - y2)) * inv_d
                    w1 = ((y2 - y0) * (fx_p - x2) + (x0 - x2) * (fy_p - y2)) * inv_d
                    w2 = 1.0 - w0 - w1
                    if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                        z = w0 * z0 + w1 * z1 + w2 * z2
                        if z < zbuf[py, px]:
                            zbuf[py, px] = z
                            fb[py, px, 0] = c_red
                            fb[py, px, 1] = 0
                            fb[py, px, 2] = 0


# ==============================================================================
# 9. INTERFEJS GRAFICZNY OKNA (PyQt5)
# ==============================================================================

class ThreeCanvas(QtWidgets.QWidget):
    def __init__(self, camera, on_run_callback):
        super().__init__()
        self.camera = camera
        self.on_run_callback = on_run_callback
        self.mouseDown = False
        self.last_pos = None
        self.fb = np.zeros((600, 800, 3), dtype=np.uint8)
        self.zbuf = np.full((600, 800), 1e9, dtype=np.float32)

    def resizeEvent(self, event):
        w, h = max(100, self.width()), max(100, self.height())
        self.fb = np.zeros((h, w, 3), dtype=np.uint8)
        self.zbuf = np.full((h, w), 1e9, dtype=np.float32)

    def mousePressEvent(self, evt):
        gGrabber.start(evt.x(), evt.y(), self.width(), self.height(), self.on_run_callback)
        self.mouseDown = True
        self.last_pos = (evt.x(), evt.y())

    def mouseMoveEvent(self, evt):
        if self.mouseDown:
            if gGrabber.is_active:
                gGrabber.move(evt.x(), evt.y(), self.width(), self.height())
            elif self.last_pos is not None:
                self.camera.theta -= (evt.x() - self.last_pos[0]) * 0.007
                self.camera.phi = float(np.clip(self.camera.phi - (evt.y() - self.last_pos[1]) * 0.007, 0.1, math.pi - 0.1))
            self.last_pos = (evt.x(), evt.y())

    def mouseReleaseEvent(self, evt):
        if gGrabber.is_active:
            gGrabber.end()
        self.mouseDown = False

    def wheelEvent(self, evt):
        delta = evt.angleDelta().y()
        if not gGrabber.adjustDistance(delta):
            factor = 0.9 if delta > 0 else 1.1
            self.camera.radius = float(np.clip(self.camera.radius * factor, 0.2, 5.0))

    def paintEvent(self, event):
        cloth = gPhysicsScene["objects"][0]
        cam_pos, cam_r, cam_u, cam_f = self.camera.basis()
        _render_three_scene(
            self.fb,
            self.zbuf,
            cloth.numParticles,
            cloth.pos,
            cloth.normals,
            cloth.faceTriIds,
            cloth.stretchingIds,
            cloth.numStretch,
            gPhysicsScene["showEdges"],
            gPhysicsScene["sphereEnabled"],
            gPhysicsScene["spherePos"],
            gPhysicsScene["sphereRadius"],
            cam_pos,
            cam_r,
            cam_u,
            cam_f,
            math.radians(self.camera.fov),
        )
        h, w, _ = self.fb.shape
        qimg = QtGui.QImage(self.fb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        QtGui.QPainter(self).drawImage(0, 0, qimg)


class ClothSimulationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        global gGrabber
        self.setWindowTitle("Cloth Simulation")
        self.resize(960, 720)

        self.camera = OrbitCamera()
        gGrabber = Grabber(self.camera)
        mesh = initPhysics()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        title = QtWidgets.QLabel("Cloth Simulation")
        title.setFont(QtGui.QFont("Verdana", 18, QtGui.QFont.Bold))
        layout.addWidget(title)

        top_row = QtWidgets.QHBoxLayout()
        btn_css = "background-color: #606060; border: none; color: white; padding: 10px 24px; font-size: 16px;"

        self.buttonRun = QtWidgets.QPushButton("Run")
        self.buttonRun.setStyleSheet(btn_css)
        self.buttonRun.clicked.connect(self.run)
        top_row.addWidget(self.buttonRun)

        btn_restart = QtWidgets.QPushButton("Restart")
        btn_restart.setStyleSheet(btn_css)
        btn_restart.clicked.connect(self.restart)
        top_row.addWidget(btn_restart)

        chk_edges = QtWidgets.QCheckBox("Show edges")
        chk_edges.clicked.connect(onShowEdges)
        top_row.addWidget(chk_edges)

        chk_wind = QtWidgets.QCheckBox("Wind")
        chk_wind.toggled.connect(lambda v: gPhysicsScene.__setitem__("windEnabled", v))
        top_row.addWidget(chk_wind)

        self.chk_sphere = QtWidgets.QCheckBox("Sphere collision")
        self.chk_sphere.toggled.connect(self.on_toggle_sphere)
        top_row.addWidget(self.chk_sphere)

        chk_tear = QtWidgets.QCheckBox("Tear cloth")
        chk_tear.toggled.connect(lambda v: gPhysicsScene.__setitem__("tearEnabled", v))
        top_row.addWidget(chk_tear)

        top_row.addStretch()
        layout.addLayout(top_row)

        self.numTris = len(mesh["faceTriIds"]) // 3
        numVerts = gPhysicsScene["objects"][0].numParticles
        self.uiFrames = 0
        self.uiFpsT0 = time.perf_counter()
        self.uiFpsDisplay = "0"
        self.stats_label = QtWidgets.QLabel(
            f"{self.numTris} tris  {numVerts} verts  0.000 ms per frame  |  0 FPS (sim)  |  0 FPS (UI)"
        )
        self.stats_label.setFont(QtGui.QFont("Verdana", 11))
        layout.addWidget(self.stats_label)

        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel("Bending compliance:"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setValue(1)
        self.slider.setFixedWidth(180)
        self.slider.valueChanged.connect(self.on_slider)
        slider_row.addWidget(self.slider)
        slider_row.addStretch()
        layout.addLayout(slider_row)

        self.canvas = ThreeCanvas(self.camera, self.run)
        layout.addWidget(self.canvas, stretch=1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(16)

    def on_slider(self, value):
        for obj in gPhysicsScene["objects"]:
            obj.bendingCompliance = float(value)

    def on_toggle_sphere(self, checked):
        """Spawnuje kulę 3D przed tkaniną; kula spada na podłogę i można ją rzucać myszką."""
        gPhysicsScene["sphereEnabled"] = checked
        gPhysicsScene["sphereGrabbed"] = False
        if checked:
            gPhysicsScene["spherePos"][:] = [0.0, 0.45, 0.28]
            gPhysicsScene["sphereVel"][:] = [0.0, 0.0, 0.0]

    def run(self):
        if gPhysicsScene["paused"]:
            self.buttonRun.setText("Stop")
        else:
            self.buttonRun.setText("Run")
        gPhysicsScene["paused"] = not gPhysicsScene["paused"]

    def restart(self):
        global timeFrames, timeSum, msDisplay, fpsDisplay
        initPhysics()
        gPhysicsScene["sphereGrabbed"] = False
        gPhysicsScene["spherePos"][:] = [0.0, 0.45, 0.28]
        gPhysicsScene["sphereVel"][:] = [0.0, 0.0, 0.0]
        timeFrames = 0
        timeSum = 0.0
        msDisplay = "0.000"
        fpsDisplay = "0"
        for obj in gPhysicsScene["objects"]:
            obj.bendingCompliance = float(self.slider.value())

    def update_loop(self):
        simulate()
        # Zliczanie rzeczywistego klatkażu okna UI (odświeżane co 0.5 s)
        self.uiFrames += 1
        now = time.perf_counter()
        elapsed = now - self.uiFpsT0
        if elapsed >= 0.5:
            self.uiFpsDisplay = f"{self.uiFrames / elapsed:.0f}"
            self.uiFrames = 0
            self.uiFpsT0 = now
        numVerts = gPhysicsScene["objects"][0].numParticles
        sim_fps = "0" if gPhysicsScene["paused"] else fpsDisplay
        self.stats_label.setText(
            f"{self.numTris} tris  {numVerts} verts  {msDisplay} ms per frame  |  {sim_fps} FPS (sim)  |  {self.uiFpsDisplay} FPS (UI)"
        )
        self.canvas.update()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ClothSimulationApp()
    win.show()
    sys.exit(app.exec_())
