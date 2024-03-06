import React, {useState, useEffect} from 'react'
import axios from 'axios'
import './ResultsCodeAbnormal.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
import {Button} from "@mui/material";
import {
	Chart as ChartJS,
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend,
  } from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
	CategoryScale,
	LinearScale,
	BarElement,
	Title,
	Tooltip,
	Legend
);


export default function ResultsCodeAbnormal () {

  const [Data, setData] = useState([]);
	const [onLoad, setonLoad] = useState(true);
  let [imageExists, setImageExists] = useState(false)
	const [images, setImages] = useState([]);
	const [gradimages, setGradImages] = useState([]);
  let [gradstate, setGradState] = useState(false);
  let [page, setPage] = useState(images);
	
	// Component가 처음 마운트 될 때 1번만 데이터를 가져옵니다
	useEffect(() => {
		const fetchData = async () => {
			try {
				const response = await axios.get("http://127.0.0.1:8000/totalresult");
				const data = response.data;
				setData(data);
				setonLoad(false);
				} catch (error) {
					console.log(error);
					setonLoad(true);
				}
		};
		fetchData();
	}, []);

  useEffect(() => {
    const fetchImage = async () => {
      try {
        const response_original = await axios.get("http://127.0.0.1:8000/result/abnormal/original");
        const response_grad = await axios.get("http://127.0.0.1:8000/result/abnormal/gradcam");
        const data_original = response_original.data;
        const data_grad = response_grad.data;
        setImages(data_original)
        setGradImages(data_grad)
        setImageExists(true);
        setonLoad(false);
      } catch (error) {
        console.log(error);
        setonLoad(true);
        setImageExists(false);
      }
    };
    fetchImage();
  }, []);

  function showGrad (e) {
		e.preventDefault();
		if (gradstate === false) {
			setGradState(true);
		} else {
			setGradState(false);
		}
		}

  useEffect(() => {
    // Set the initial page based on gradstate
    setPage(gradstate ? gradimages : images);
    }, [gradstate, images, gradimages]);

  const handleImageError = () => {
		setImageExists(false); // Set imageExists to false if the image fails to load
	  };

  const aboptions = {
		indexAxis: 'y',
		elements: {
		  bar: {
			borderWidth: 0.5,
		  },
		},
		aspectRatio: 12,
    // maintainAspectRatio: false,
		responsive: true,
		plugins: {
		  legend: {
      display: false,
			position: 'right',
		  },
		  title: {
			display: false,
			text: 'Results',
		  },
		},
    scales: {
      // to remove the labels
      x: {
        min:0,
        max:100,
        ticks: {
          // display: false,
        },
  
        // to remove the x-axis grid
        grid: {
          drawBorder: false,
          display: false,
        },
      },
      // to remove the y-axis labels
      y: {
        ticks: {
          display: false,
          beginAtZero: true,
        },
        // to remove the y-axis grid
        grid: {
          drawBorder: false,
          display: false,
        },
      },
    },
	  };

    const acloptions = {
      indexAxis: 'y',
      elements: {
        bar: {
        borderWidth: 0.5,
        },
      },
      aspectRatio: 10,
      maintainAspectRatio: false,
      responsive: true,
      plugins: {
        legend: {
        display: false,
        position: 'right',
        },
        title: {
        display: false,
        text: 'Results',
        },
      },
      scales: {
        // to remove the labels
        x: {
          min: 0,
          max: 100,
          ticks: {
            // display: false,
          },
    
          // to remove the x-axis grid
          grid: {
            drawBorder: false,
            display: false,
          },
        },
        // to remove the y-axis labels
        y: {
          ticks: {
            display: false,
            beginAtZero: true,
          },
          // to remove the y-axis grid
          grid: {
            drawBorder: false,
            display: false,
          },
        },
      },
      };

	let labels = [];
  if (Data.labels) {
      labels = Data.labels;
  }

  let abnormaldata = [];
  let acldata = [];
  let meniscusdata = [];
  let totalresult = [0, 0, 0, 0];
  if (Data.datasets) {
      abnormaldata = Data.datasets[0];
      acldata = Data.datasets[1];
      meniscusdata = Data.datasets[2];
      let isabnormal = abnormaldata.x > 50;
      let isacl = acldata.x > 50;
      let ismeniscus = meniscusdata.x > 50;
      if (isabnormal) {
        if (isacl) {
          totalresult[1] += 1;
        }
        if (ismeniscus) {
          totalresult[2] += 1;
        }
        if (!isacl && !ismeniscus) {
          totalresult = [0, 0, 0, 1];
        }
      } else {
        totalresult = [1, 0, 0, 0];
      }
  }

  const abnormaldataset = {
      labels,
      datasets: [
          {
              label: "%",
              data: [abnormaldata],
              borderColor: "rgb(255, 99, 132)",
              backgroundColor: "rgba(255, 99, 132, 0.5)",
              barThickness: 40
          },
      ],
  };
  const acldataset = {
    labels,
    datasets: [
        {
            label: "%",
            data: [acldata],
            borderColor: "rgb(255, 99, 132)",
            backgroundColor: "rgba(255, 99, 132, 0.5)",
            barThickness: 40
        },
    ],
  };
  const meniscusdataset = {
    labels,
    datasets: [
        {
            label: "%",
            data: [meniscusdata],
            borderColor: "rgb(255, 99, 132)",
            backgroundColor: "rgba(255, 99, 132, 0.5)",
            barThickness: 40
        },
    ],
};

	return (
		<div className="results-abnormal">
      <div className="div">
        <div className="top-overlay">
          <div className="overlap-group">
            <div className="overlap">
              <div className="text-wrapper">환자명: 김00</div>
              <div className="text-wrapper-2">성별: F</div>
              <div className="text-wrapper-3">검사일: 2024-02-27</div>
            </div>
            <Link to="/">
            <img className="logo" alt="Logo" src={ImgAsset.ResultsCodeACL_logo} />
            </Link>
          </div>
        </div>
        <div className="overlap-2">
          <div className="abnormal-score">
            <div className="overlap-3">
              <div className="text-wrapper-4">Abnormality Score</div>
              <div className="rectangle">
                <Bar options={aboptions} data={abnormaldataset} />
              </div>
              <div className="text-wrapper-5">{onLoad ? '?' : `${abnormaldataset.datasets[0].data[0].x}`}</div>
              <div className="text-wrapper-6">%</div>
            </div>
          </div>
          <div className="category">
            <div className="text-wrapper-7">Category</div>
            <div className='switch'>
              <Button variant="contained" onClick={showGrad} sx={{ color: 'white', backgroundColor: 'black' }}>
                Inspect
              </Button>
            </div>
            {/* <img className="switch" alt="Switch" src="switch.png" /> */}
            {/* <div className="text-wrapper-8">Show</div> */}
            <div className="abnormal-button">
              <div className="overlap-group-2">
                <div className="abnormal-cat" />
                <div className="text-wrapper-9">Abnormal</div>
              </div>
            </div>
            <div className="meniscus-button">
              <div className="overlap-group-2">
                <div className="meniscus-cat" />
                <div className="text-wrapper-10">Meniscus</div>
              </div>
            </div>
            <div className="ACL-button">
              <div className="div-wrapper">
                <div className="text-wrapper-11">ACL</div>
              </div>
            </div>
          </div>
        </div>
        <div className="ACL-score">
          <div className="overlap-4">
            <div className="text-wrapper-12">ACL Score</div>
            <div className="overlap-5">
              <div className="rectangle-2">
                <Bar options={acloptions} data={acldataset} />
              </div>
              <div className="text-wrapper-13">{onLoad ? '?' : `${acldataset.datasets[0].data[0].x}`}</div>
              <div className="text-wrapper-14">%</div>
            </div>
          </div>
        </div>
        <div className="meniscus-score">
          <div className="overlap-6">
            <div className="text-wrapper-15">Meniscus Score</div>
            <div className="overlap-7">
              <div className="rectangle-0">
                <Bar options={acloptions} data={meniscusdataset} />
              </div>
              <div className="text-wrapper-13">{onLoad ? '?' : `${meniscusdataset.datasets[0].data[0].x}`}</div>
              <div className="text-wrapper-14">%</div>
            </div>
          </div>
        </div>
        <div className="result">
          <div className="text-wrapper-16">Result</div>
          <div className="overlap-8" style={!totalresult[0] ? {backgroundColor:'#3c3c3c'} : {backgroundColor:'#ff7171'}}>
            <div className="text-wrapper-17">Normal</div>
          </div>
          <div className="overlap-9" style={!totalresult[3] ? {backgroundColor:'#3c3c3c'} : {backgroundColor:'#ff7171'}}>
            <div className="text-wrapper-17">Others</div>
          </div>
          <div className="overlap-10" style={!totalresult[2] ? {backgroundColor:'#3c3c3c'} : {backgroundColor:'#ff7171'}}>
            <div className="text-wrapper-18">Meniscus</div>
          </div>
          <div className="overlap-11" style={!totalresult[1] ? {backgroundColor:'#3c3c3c'} : {backgroundColor:'#ff7171'}}>
            <div className="text-wrapper-19">ACL</div>
          </div>
        </div>
        <div className="axial-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
              {/* <div className="overlap-group-3"> */}
              <div className='overlap-group-3'>
                {onLoad ? (
                    <span>Loading...</span>
                  ) : imageExists ? (
                    page[0] ? (
                      <img
                        className="overlap-group-3"
                        alt="Press Inspect to Start!"
                        key='0'
                        src={`data:image/png;base64,${page[0].body}`}
                        onError={handleImageError}
                      />
                    ) : (
                      <span>Image not found</span>
                    )
                  ) : (
                    <span>Error loading images</span>
                  )}
                </div>
                {/* <div className="graph-text">Axial</div> */}
                <div className="heatmap-bar">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-20">0</div>
                  <div className="text-wrapper-21">100</div>
                </div>
              {/* </div> */}
            </div>
            <Link to="/axialresults">
            <div className="inspect-ax">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            </Link>
            <div className="plus-ax">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div>
          </div>
        </div>
        <div className="coronal-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
              {/* <div className="overlap-group-3"> */}
                {/* <div className="graph-text">Coronal</div> */}
                <div className='overlap-group-3'>
                {onLoad ? (
                    <span>Loading...</span>
                  ) : imageExists ? (
                    page[1] ? (
                      <img
                        className="overlap-group-3"
                        alt="Press Inspect to Start!"
                        key='1'
                        src={`data:image/png;base64,${page[1].body}`}
                        onError={handleImageError}
                      />
                    ) : (
                      <span>Image not found</span>
                    )
                  ) : (
                    <span>Error loading images</span>
                  )}
                </div>
                <div className="heatmap-bar-2">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-20">0</div>
                  <div className="text-wrapper-21">100</div>
                </div>
              {/* </div> */}
            </div>
            <div className="inspect-co">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            <div className="plus-co">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div>
          </div>
        </div>
        <div className="sagittal-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
              {/* <div className="overlap-group-3"> */}
                {/* <div className="graph-text">Sagittal</div> */}
                <div className='overlap-group-3'>
                {onLoad ? (
                    <span>Loading...</span>
                  ) : imageExists ? (
                    page[2] ? (
                      <img
                        className="overlap-group-3"
                        alt="Press Inspect to Start!"
                        key='2'
                        src={`data:image/png;base64,${page[2].body}`}
                        onError={handleImageError}
                      />
                    ) : (
                      <span>Image not found</span>
                    )
                  ) : (
                    <span>Error loading images</span>
                  )}
                </div>
                <div className="heatmap-bar-3">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-20">0</div>
                  <div className="text-wrapper-21">100</div>
                </div>
              {/* </div> */}
            </div>
            <div className="inspect-sag">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            <div className="plus-sag">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
	)
}