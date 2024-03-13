# Modelos de Recuperación de Información

## Autores

Amananda Noris Hernández

Juan Miguel Pérez Martínez

Marcos Antonio Pérez Lorenzo

## Definición de los modelos de SRI implementados

Nuestro programa implementa 3 modelos de recuperación de información, el Modelo Booleano(MRIB), el Modelo Booleano Extendido y el Modelo de Indexación Semántica Latente.

### Modelo Booleano

El MRIB está basado en la Lógica Booleana y la clásica Teoría de Conjuntos en el cual ambos, los documentos a buscar y la consulta del usuario, son concebidos como un conjunto de términos.La recuperación está basada en cuando los documentos contienen o no los términos de la consulta. Dado un conjunto finito

    T = {t1, t2, ..., tj, ..., tm}

de elementos llamados índices (e.g. palabras o expresiones - las cuales pueden estar lematizadas - describiendo o caracterizando documentos como son palabras dadas para un artículo de un periódico ), un conjunto finito

    D = {D1, ..., Di, ..., Dn}, donde Di es un elemento del conjunto potencia de T

de elementos llamados documentos.Dada una expresión Booleana - en forma normal - Q llamada consulta como sigue a continuación:

    Q = (Wi OR Wk OR ...) AND ... AND (Wj OR Ws OR ...) ,
    con Wi=ti, Wk=tk, Wj=tj, Ws=ts, or Wi=NON ti, Wk=NON tk, Wj=NON tj, Ws=NON ts

donde ti significa que el término ti está presente en el documento Di y, por el contrario, NON ti significa que no está.

Equivalentemente, Q puede ser dado en forma normal disjuntiva, también.Una operación de recuperación consiste de dos pasos como se define a continuación:

    1. El conjunto Sj de documentos que son obtenidos que contienen o no el término tj (dependiendo de cuando Wj=tj o Wj=NON tj) :

        Sj = {Di | Wj elemento de Di}

    2. Estos documentos son recuperados como respuesta a Q, los cuales son el resultado de las correspondientes operaciones entre conjuntos, i.e. la respuesta a Q es como sigue:

        UNION ( INTERSECCION Sj)

### Modelo Booleano Extendido

En el Modelo Booleano Extendido un documento se representa por un vector (al igual que en el Modelo Vectorial). Cada componente corresponde a un término asociado al documento.

El peso del término K x {\displaystyle K_{x}} asociado al documento d j {\displaystyle d_{j}} se mide por su frecuencia de término normalizada y puede definirse como:

w x , j = f x , j ∗ I d f x m a x i I d f x {\displaystyle w_{x,j}=f_{x,j}*{\frac {Idf_{x}}{max_{i}Idf_{x}}}}

donde I d f x {\displaystyle Idf_{x}} es la frecuencia inversa de documento.

El vector de pesos asociado al documento d j {\displaystyle d_{j}} puede ser representado como:

v d j = [ w 1 , j , w 2 , j , … , w i , j ] {\displaystyle \mathbf {v} _{d_{j}}=[w_{1,j},w_{2,j},\ldots ,w_{i,j}]}

### Modelo de Indexación Semántica Latente

El modelo de recuperación de información de indexación semántica latente (LSI, por sus siglas en inglés) se basa en el cuádruplo [D, Q, F, R(aj, d;)] y se puede describir de la siguiente manera:

D: En el contexto de LSI, los documentos se representan en un espacio de alta dimensión, donde cada dimensión corresponde a un término en el corpus de documentos. La representación de un documento en este espacio se obtiene a través de una descomposición en valores singulares (SVD) de la matriz de documentos-términos, que reduce la dimensionalidad de los datos y revela las relaciones semánticas latentes entre los términos y los documentos.

Q: Es un conjunto de consultas, donde cada consulta se representa mediante una representación lógica que se asemeja a la de los documentos en el espacio de alta dimensión. Las consultas se mapean en el espacio LSI para calcular la similitud entre ellas y los documentos. La representación de una consulta en este espacio se obtiene de manera similar a la de los documentos, utilizando la matriz de consultas-términos y aplicando una SVD mediante:

qk = Ek-1UkTq

F: En LSI, el framework que modela las representaciones de los documentos, consultas y sus relaciones se basa en el modelo de espacio vectorial, donde la similitud entre una consulta y un documento se mide como el coseno del ángulo entre sus representaciones en el espacio LSI. Este modelo permite identificar documentos relevantes para una consulta basándose en la similitud semántica de los términos utilizados en la consulta y en los documentos.

R(aj, d;): La evaluación de esta función establece un cierto orden entre los documentos de acuerdo a la consulta, utilizando la similitud semántica calculada en el espacio LSI. Los documentos se clasifican en función de su relevancia para la consulta, de manera que los más relevantes aparezcan primero en la lista de resultados.

En resumen, el modelo de indexación semántica latente utiliza la descomposición en valores singulares para reducir la dimensionalidad de los datos de documentos y consultas, permitiendo identificar relaciones semánticas latentes entre ellos. Este enfoque mejora la eficiencia de la recuperación de información al revelar la similitud semántica entre términos y documentos, incluso cuando sus perfiles de términos son diferentes.

## Consideraciones tomadas a la hora de desarrollar la solución. Definición de la consulta

Se tuvo en cuenta la definición de ambos modelos de recuperación de información, así como las métricas extistentes para la evaluación de los mismos. El corpus se tomó de ir-dataset "beir/nfcorpus/test" ya que provee documentos, consultas y las respuetas a dichas consultas.

La consulta se obtiene en lenguaje natural según lo orientado y se define de acuerdo al modelo implementado.

## Explicación de cómo ejecutar el proyecto. Posibles entradas de parámetros

Para ejecutar el proyecto basta correr en wsl /startup.sh y en windows, con ejecutar el archivo main.py .

## Explicación de la solución desarrollada

La solución desarrollada es un recuperador de información que permite al usuario, a través de una página web, hacer consultas a una base de datos y recibir las respuestas brindadas por varios modelos, pudiendo así comparar la calidad de los mismos. Además se le permite marcar cuáles documentos considera relevantes para una posterior retroalimentación del programa y se le brindan recomendaciones sobre qué otros documentos pudieran ser de su interés. Nuestro programa cuenta con módulos para:

1. **downloader**: Importa el módulo nltk y define una función llamada program que, cuando se ejecuta, abre la interfaz gráfica de usuario (GUI) del descargador de NLTK. Esta GUI permite al usuario seleccionar y descargar corpora, modelos y otros paquetes de datos que pueden ser utilizados con NLTK. La función nltk.download_gui() es responsable de iniciar esta interfaz, proporcionando una forma interactiva para los usuarios de descargar y gestionar los recursos necesarios para trabajar con NLTK.

2. **loader**: Define una clase Loader que se encarga de cargar y gestionar diferentes modelos de procesamiento de lenguaje natural (NLP) y sus componentes asociados, como vocabularios, vectores de documentos y vectorizadores. La clase Loader proporciona métodos para cargar el modelo booleano, booleano extendido y LSI, así como para cargar y guardar corpus, vocabularios, vectorizadores y documentos relevantes. Además, incluye métodos para realizar búsquedas utilizando estos modelos y para guardar y cargar documentos relevantes. La clase DefaultLoader hereda de Loader y proporciona una implementación predeterminada para cargar y gestionar estos componentes, incluyendo la carga perezosa de los modelos y componentes necesarios.

3. **models**: Implementa un sistema de búsqueda de texto basado en modelos de similitud de documentos utilizando técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático. Comienza importando las bibliotecas necesarias para el procesamiento de texto, la creación de vectores de documentos, y la medición de similitud entre ellos. Define una estructura de datos BooleanQuery para almacenar consultas booleanas y clases Model, BooleanModel, ExtendedModel, y LSIModel para manejar diferentes tipos de modelos de búsqueda. Model es la clase base que inicializa el preprocesamiento de datos. BooleanModel extiende Model para manejar consultas booleanas, incluyendo la tokenización, eliminación de ruido, y reducción morfológica. También implementa la creación de consultas booleanas y la similitud entre consultas y documentos. ExtendedModel mejora BooleanModel al calcular la similitud como un valor flotante en lugar de binario, lo que permite una evaluación más precisa de la similitud. LSIModel utiliza el Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de los vectores de documentos y mejorar la eficiencia de la búsqueda. Implementa la creación de consultas y la búsqueda de documentos relevantes utilizando la similitud del coseno.

4. **preprocessing**: Define una clase Preprocessing que realiza varias tareas de preprocesamiento de texto utilizando la biblioteca NLTK y otras herramientas de Python. Primero, tokeniza el texto, luego elimina ruido y palabras vacías, y finalmente realiza una reducción morfológica mediante lematización o stemming. Además, se definen dos subclases, BooleanPreprocessing y VectorPreprocessing, que extienden la funcionalidad de la clase base para crear representaciones booleanas y vectoriales de los textos, respectivamente. También incluye la carga de un conjunto de datos, el preprocesamiento de este conjunto de datos utilizando ambas subclases, y la serialización de los resultados y el vocabulario para su posterior uso.

5. **recommendation**: Define una clase Recommendation que utiliza el corpus WordNet de NLTK para encontrar sinónimos de palabras en un texto objetivo, y luego utiliza el algoritmo K-Nearest Neighbors (KNN) de scikit-learn para recomendar documentos similares basados en la similitud semántica. Primero, inicializa la clase con un conjunto de datos, un vectorizador y documentos vectorizados. Luego, en el método recommend, procesa el texto objetivo para encontrar sus sinónimos utilizando WordNet, los combina en un nuevo texto objetivo, y lo transforma en un vector utilizando el vectorizador. Finalmente, utiliza el modelo KNN entrenado para encontrar los índices de los documentos más cercanos en el conjunto de datos, basándose en la similitud semántica del nuevo texto objetivo.

6. **test**: Define una función programs que realiza una evaluación de modelos de búsqueda de texto utilizando diferentes métodos de búsqueda (boolean, extended, lsi) sobre un conjunto de datos de corpus booleanos. Primero, carga el conjunto de datos y los documentos correspondientes utilizando un DefaultLoader. Luego, para cada consulta en el conjunto de datos, ejecuta cada método de búsqueda y compara los resultados con los documentos esperados (definidos en qrels_iter). Los resultados de la búsqueda se almacenan en matrices binarias (y_preds y y_trues) que representan si un documento fue recuperado o no para cada consulta y método de búsqueda. Finalmente, calcula y muestra métricas de evaluación como precisión, recall, F1-score y fallout para cada método de búsqueda, utilizando las matrices de confusión multietiqueta y las funciones de precisión, recall y F1-score de sklearn.metrics.

7. **app**: Es una aplicación de Streamlit que utiliza la biblioteca streamlit-aggrid para mostrar datos en una interfaz de usuario interactiva. La aplicación permite a los usuarios realizar búsquedas de texto y recibir recomendaciones basadas en los resultados de la búsqueda. Utiliza un cargador de datos personalizado (DefaultLoader) para buscar coincidencias de texto y recomendaciones, y luego muestra los resultados en cuadrículas interactivas. Los usuarios pueden marcar documentos como relevantes, y estos se guardan para futuras referencias.

En cuanto a las definiciones matemáticas y algorítmicas consideramos:

-**k-Nearest Neighbors**: El algoritmo k-Nearest Neighbors (k-NN) es un método de aprendizaje supervisado utilizado para la clasificación y la regresión. Su funcionamiento se basa en el principio de que los puntos de datos que están cerca uno del otro en el espacio de características pertenecen a la misma clase. En el contexto de las recomendaciones, k-NN se utiliza para sugerir elementos similares a los que un usuario ha interactuado previamente.

-**Similitud cosénica**: La similitud del coseno es una medida de similitud entre dos vectores no nulos definidos en un espacio de producto interno. Se calcula como el coseno del ángulo entre los dos vectores, es decir, es el producto punto de los vectores dividido por el producto de sus longitudes. Esto significa que la similitud del coseno no depende de las magnitudes de los vectores, sino solo del ángulo entre ellos. La similitud del coseno siempre pertenece al intervalo [-1, 1].

-**Descomposición en Valores Singulares**: La SVD, por sus siglas en inglés, es una factorización de una matriz real o compleja en tres matrices. Esta descomposición es fundamental en álgebra lineal y tiene aplicaciones en diversas áreas, incluyendo procesamiento de señales, imágenes y big data 2. La SVD de una matriz (A) se representa como (A = U\Sigma V^T), donde:

(U) es una matriz ortogonal de dimensiones (m \times m) que contiene los vectores singulares izquierdos de (A).
(\Sigma) es una matriz diagonal de dimensiones (m \times n) que contiene los valores singulares de (A) en su diagonal principal, ordenados de mayor a menor.
(V^T) es la transpuesta de una matriz ortogonal de dimensiones (n \times n) que contiene los vectores singulares derechos de (A).

Los valores singulares son las raíces cuadradas de los valores propios de (A^TA) o (AA^T), dependiendo de si (A) es de dimensiones (m \times n) o (n \times m). Los vectores singulares izquierdos y derechos son los vectores propios de (A^TA) y (AA^T) correspondientes a estos valores singulares 35.

-**Vectorización TF-IDF**: El vectorizador TF-IDF (Term Frequency-Inverse Document Frequency) convierte el texto en un vector de números reales. Cada número representa la importancia de una palabra en el texto. Las palabras que aparecen con frecuencia en el texto pero raramente en otros documentos tienen un número más alto.

-**FNC**: En lógica booleana, una fórmula está en forma normal conjuntiva (FNC) si corresponde a una conjunción de cláusulas, donde una cláusula es una disyunción de literales, donde un literal y su complemento no pueden aparecer en la misma cláusula.

-**FND**: En lógica booleana, una forma normal disyuntiva (FND) es una estandarización (o normalización) de una fórmula lógica que es una disyunción de cláusulas conjuntivas. Una fórmula FND está en forma normal disyuntiva completa si cada una de sus variables aparece exactamente una vez en cada cláusula. 

## Insuficiencias de la solución y mejoras propuestas

-**Demora en el procesamiento**: Para conjuntos de datos de gran tamaño nuestro programa puede llegar a demorar un poco a la hora de cargar el corpus o realizar algunas consultas.

### Mejoras propuestas

-**Paralelizacion de procesos**: Proponemos la paralelización de algunos procesos/funciones del programa para aumentar la rapidez de las respuetas, sin embargo se debe tener en cuenta que para realizar dichas acciones es necesario, en la mayoría de los casos, un gran poder de cómputo.
