

## Niveau intermédiaire : tabulaire

1. **Taxi-v3 avec Q-Learning**

   * Implémente un Q-Table (`np.zeros([n_states, n_actions])`).
   * Utilise epsilon-greedy pour l’exploration.
   * Trace la courbe de reward moyen sur 5000 épisodes.
     Objectif : que ton agent atteigne plus de 90% de réussite.

2. **FrozenLake-v1 avec SARSA**

   * Implémente SARSA (on-policy).
   * Compare les résultats avec Q-Learning.
     Objectif : observer la différence de stabilité entre les deux.

3. **CliffWalking-v0**

   * Implémente Q-Learning.
   * Affiche la *policy finale* sous forme de grille (flèches pour actions).
     Objectif : apprendre à éviter la falaise.

---

## Niveau avancé : approximation de fonctions

4. **CartPole-v1 avec DQN**

   * Implémente un petit réseau de neurones (2 couches denses).
   * Ajoute replay buffer + target network.
     Objectif : atteindre un score supérieur à 195 sur 100 épisodes consécutifs.

5. **LunarLander-v2 avec Policy Gradient (REINFORCE)**

   * Réseau qui sort une distribution de probabilités (Softmax sur actions).
   * Mise à jour par gradient de politique.
     Objectif : réussir à atterrir sans crash la majorité du temps.

6. **Acrobot-v1 avec Actor-Critic**

   * Implémente un réseau partagé (features) puis deux têtes : policy et value.
   * Teste la vitesse de convergence par rapport à REINFORCE.
     Objectif : balancer l’acrobot au-dessus du seuil.

---

## Bonus avancé

7. **LunarLander-v2 avec PPO (Stable-Baselines3)**

   * Entraîne avec `PPO("MlpPolicy", env)`.
   * Compare les learning curves avec ton REINFORCE maison.

8. **Breakout-v4 (Atari)**

   * Prétraite les frames (resize, grayscale).
   * Lance un agent PPO de SB3.
     Objectif : dépasser un score de 50.

