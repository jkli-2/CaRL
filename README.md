<p align="center" style="font-size:40px;">
<b> CaRL: Learning Scalable Planning Policies with Simple Rewards </b>
</p>

<h3 align="center">
         <a href="https://arxiv.org/abs/2504.17838"> Paper</a>
</h3>

<p align="center" style="font-size:20px;">
This repository contains the first public codebase for doing Reinforcement Learning with the CARLA leaderboard 2.0 and nuPlan. We provide pre-trained model weights for CaRL which is the best open-source RL planner on longest6 v2 and nuPlan. Additionally, we provide reproductions of the popular CARLA planners Roach, PlanT and Think2Drive for the CARLA leaderboard 2.0. <br/><br/>
</p>

The code and documentation for the respective simulators can be found in the [CARLA folder](CARLA) and the [nuPlan_folder](nuPlan).


## License

The original code in this repository is provided under the Civil-M license, which is a variant of the MIT license that bans dual-use. [The license](LICENSE) contains a partial copyleft which requires derivative work to include the civil clause in their license. For further information see the accompaning documentation on [Civil Software Licenses](docs/Jaeger2025LicenseWhitepaper.pdf).

## Citation
If you find the repo useful, please consider giving it a star &#127775;.
To cite the paper please use the following bibtex:
```BibTeX
@article{Jaeger2025ArXiv, 
        author = {Bernhard Jaeger and Daniel Dauner and Jens Beißwenger and Simon Gerstenecker and Kashyap Chitta and Andreas Geiger}, 
        title = {CaRL: Learning Scalable Planning Policies with Simple Rewards}, 
        year = {2025}, 
        journal = {arXiv.org}, 
        volume = {2504.17838}, 
}
```

## Acknowledgements
The original code in this repository was written by Bernhard Jaeger, Daniel Dauner, Jens Beißwenger and Simon Gerstenecker. Kashyap Chitta and Andreas Geiger  have contributed as technical advisors.

Open source code like this is build on the shoulders of many other open source repositories.
Particularly, we would like to thank the following repositories for their contributions:

* [leaderboard](https://github.com/carla-simulator/leaderboard)
* [scenario_runner](https://github.com/carla-simulator/scenario_runner)
* [carla_garage](https://github.com/autonomousvision/carla_garage)
* [plant](https://github.com/autonomousvision/plant)
* [roach](https://github.com/zhejz/carla-roach/)
* [tuplan_garage](https://github.com/autonomousvision/tuplan_garage)

We also thank the creators of the numerous libraries we use. Complex projects like this would not be feasible without your contribution.
