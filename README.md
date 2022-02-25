#### 알고리즘 설계기법

<br><br>

#### 완전탐색 Brute Force

> 반드시 답을 찾을 수가 있다
>
> 하지만 시간이 많이걸려 문제에서 요구하는 시간안에 해결할 수 있는지 고민

<br><br>

#### 탐욕기법 Gredy

> 근시안적인 판단, 뒤를 돌아보지 않아 속도가 빠르다
>
> 하지만 가설이 틀리다면 답을 못찾을 수도 있음
>
> 항상 검증이 필요

<br><br>

#### 백트래킹 BackTracking

> 기본적으로 브루트포스 방식을 사용해서 문제를 풀이
>
> 답이 될만한지 판단하고 그렇지 않으면 그 부분까지 탐색하는 것을 하지 않고 Pruning 진행
>
> 주로 DFS를 통해 모든 경우의 수를 탐색하는 과정에서 조건으로 답이 절대 될 수 없는 상황을 체크 후 이전으로 돌아가서 다시 다른 경우를 탐색하게끔 구현

<br>

백트래킹과 완전탐색(DFS)과의 차이

> - 특정 노드에서 출발하는 경로가 해결책으로 이어질 것 같지 않으면 더 이상 그 경로를 따라가지 않음으로써 시도의 횟수를 줄인다.(Pruning)
> - 완전 탐색이 모든 경로를 추적하는데 비해 백트래킹은 불필요한 경로를 조기에 차단
> - 완전탐색을 가하기에는 경우의 수가 너무나 많다.
> - **백트래킹 알고리즘을 적용하면 일반적으로 경우의 수가 줄어들지만 이 역시 최악의 경우에는 여전히 지수함수시간(Exponential Time)을 요하므로 처리 불가능할 수 있다.**



<br><br>



#### 분할정복 Divide & Conquer

<br><br>



#### 동적계획법 Dynamic Programming



<br><br>



#### 자료구조 활용

선형자료구조 : stack, queue, 

비선형 자료구조 : 용어, 저장방법, 순회(DFS,BFS)













<br><br>

#### Minimum Spanning Tree(MST)

> - 그래프에서 최소 비용 문제
>
> 모든 정점을 연결하는 간선들의 가중치의 합이 최소가 되는 트리
>
> 두 정점 사이의 최소 비용의 경로 찾기
>
> - 신장트리
>
> n개의 정점으로 이루어진 무향그래프에서 n개의 정점과 n-1개의 간선으로 이루어진 트리
>
> - 최소 신장 트리(Minimum Spanning Tree)
>
> 무향 가중치 그래프에서 신장 트리를 구성하는 간선들의 가중치의 합이 최소인 신장트리





<br><br>

#### Kruskal 알고리즘

> 간선을 하나씩 선택해서 MST를 찾는 알고리즘
>
> 1. 모든 간선을 가중치에 따라 **오름차순** 정렬
>
> 2. 가중치를 가장 낮은 간선부터 선택하면서 트리를 증가시킴
>
>    -> **사이클이 존재**하면 다음으로 가중치가 낮은 간선 선택
>
> 3. n-1개의 간선이 선택될 때 까지 반복





<br><br>

#### Prim 알고리즘

> **정점 중심**으로 해결
>
> 하나의 정점에서 연결된 간선들 중에 하나씩 선택하면서 MST를 만들어 가는 방식
>
> ​	임의의 정점을 하나 선택해서 시작
>
> ​	선택한 정점과 인접하는 정점들 중의 **최소 비용의 간선이 존재하는 정점을 선택**
>
> ​	모든 정점이 선택될 때 까지 1,2 과정을 반복

```java
    private static ArrayList<Edge>[] map;
    private static PriorityQueue<Edge> pq;
    private static boolean[] visited;
    static class Edge{
      int node, distance;

      public Edge(int node, int distance) {
        super();
        this.node = node;
        this.distance = distance;
      }		
    };

    pq = new PriorityQueue<Edge>(new Comparator<Edge>() {
      @Override
      public int compare(Edge o1, Edge o2) {
        return o1.distance-o2.distance;
			}
		});		
		
		pq.offer(new Edge(0,0));
		int result = 0;
		int maxDistance = 0;
		while(!pq.isEmpty()) {
			Edge edge = pq.poll();
			if(visited[edge.node]) continue;
			visited[edge.node] = true;
      
			result += edge.distance;
			for (Edge e : map[edge.node]) {
				if(!visited[e.node]) pq.offer(e);
			}
		}
```



<br><br>

#### 최단경로 알고리즘

> 간선의 가중치가 있는 그래프에서 두 정점 사이의 경로들 중에 간선의 가중치의 합이 최소인 경로
>
> **하나의 시작 정점**에서 끝 정점까지의 최단경로
>
> - **다익스트라**(Dijkstra) 알고리즘 : 음의 가중치를 허용X
> - **벨만-포드**(Bellman-Ford) 알고리즘 : 음의 가중치 허용
>
> 모든 정점들에 대한 최단 경로
>
> - **플로이드-워샬**(Floyd-Warshall) 알고리즘



<br><br>

#### Dijkstra 알고리즘

> **시작 정점**에서 **다른 모든 정점**으로의 최단경로를 구하는 알고리즘
>
> 시작 정점에서의 거리가 최소인 정점을 선택해 나가면서 최단경로를 구하는 방식
>
> **탐욕 기법**을 사용한 알고리즘으로 MST의 **Prim 알고리즘과 유사**

<br>

#### Code

##### 1. 인접행렬을 사용하여 구현

```java
	// 1. 인접 행렬을 사용해서 구현	
		int[] distance = new int[V]; // 출발지에서 자신으로 오는 최소비용
		boolean[] visited = new boolean[V]; // 최소비용 확정여부
		int[][] adjMatrix = new int[V][V];
		Arrays.fill(distance, Integer.MAX_VALUE);
		distance[start] = 0; // 시작점 0으로
	

		for (int i = 0; i < V; i++) {
			// 단계1 : 최소비용이 확정되지 않은 정점중 최소비용의 정점 선택
			int min=Integer.MAX_VALUE,current=0;
			for (int j = 0; j < V; j++) {
				if(!visited[j] && min>distance[j]) {
					min = distance[j];
					current = j;
				}
			}
			
			visited[current] = true;
			
			// 단계2 :  선택된 정점을 경유지로 하여 아직 최소비용이 확정되지 않은 다른정점의 최소비용을 고려
			for (int j = 0; j < V; j++) {
				if (!visited[j] && adjMatrix[current][j] != 0 && 
						distance[j] > distance[current] + adjMatrix[current][j]) {
					distance[j] =  distance[current] + adjMatrix[current][j];
				}
			}
		}
```



##### 2. 우선순위 큐를 사용해서 구현

```java
	static void dijkstra(int start) {
		
		
		int[] distance = new int[V];
		boolean[] visited = new boolean[V];
		PriorityQueue<Node> pq = new PriorityQueue<>();
		
		Arrays.fill(distance, Integer.MAX_VALUE);
		distance[start] = 0;
		pq.offer(new Node(start,distance[start]));
		
		
		while(!pq.isEmpty()) {
			
			Node current = pq.poll();
			if(visited[current.no]) continue;
			visited[current.no] = true;

			for (int j = 0; j < V; j++) {
				if(!visited[j] && adjMatrix[current.no][j]!=0 && 
						distance[j] > distance[current.no] + adjMatrix[current.no][j]) {
					distance[j] = distance[current.no] + adjMatrix[current.no][j];
					pq.offer(new Node(j,distance[j]));
				}
			}
		}
```







<br><br>



#### 플로이드 와샬 알고리즘

> - 모든 정점에서 모든 정점으로의 최단 경로 구하는 알고리즘.
>   - 다익스트라 알고리즘
>     : 하나의 정점에서 다른 모든 정점으로의 최단 경로 구하는 알고리즘
> - 거쳐가는 정점을 기준으로 최단 거리 구한는 것.

<br><br>
