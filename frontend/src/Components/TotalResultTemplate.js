import React, {useState, useEffect} from 'react'
import axios from 'axios'
import './TotalResultTemplate.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
import {Button} from "@mui/material";
import Slider from '@mui/material/Slider';
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


export default function ResultsCodeAbnormal (props) {

  const [Data, setData] = useState([]);
	const [onLoad, setonLoad] = useState(true);
  const [onPatientLoad, setonPatientLoad] = useState(true);
  let [imageExists, setImageExists] = useState(false)
	const [images, setImages] = useState([]);
	const [gradimages, setGradImages] = useState([]);
  let [gradstate, setGradState] = useState(false);
  let [page, setPage] = useState(images);
  const [patientInfo, setPatientInfo] = useState([]);
  const [patientLabel, setPatientLabel] = useState([]);
  let [gradthres, setGradThres] = useState(50);
	
  const disease = props.disease;
  const idx = props.idx;
  const ip = localStorage.getItem('userIP') || props.ip;
  const graphurl = "/result";
  const originalurl = `/result/${disease}/original`;
  const gradcamurl = `/result/${disease}/gradcam`;
  const patienturl = "/result/patient";
  const unique = Date.now();
  const exporturl = `/result/docs?cache=${unique}`;
  const config = {
    headers: {
      'IP': ip,
    }
  };

  let styleList = [0, 0, 0];
  styleList[idx] += 1;
  const whiteCircleStyle = { backgroundColor: '#ffffff' };
  const blackTextStyle = { color: '#000000' };
  const blackCircleStyle = { backgroundColor: '#3c3c3c' };
  const whiteTextStyle = { color: '#ffffff' };

  const marks = [
    {
      value: 0,
    },
    {
      value: 20,
    },
    {
      value: 50,
    },
    {
      value: 80,
    },
  ];

  function sliderChange(e) {
    e.preventDefault();
    setGradThres(e.target.value);
  }

  const exportDocs = async () => {
    try {
      const response = await axios.get(exporturl, { headers: { 'IP': ip }, responseType: 'blob' });
      // create download link
      const tempURL = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = tempURL;
      link.setAttribute('download', 'Report.docx');
      // click link and download
      document.body.appendChild(link);
      link.click();
      // delete link after download
      link.parentNode.removeChild(link);
    } catch (error) {
      alert(`리포트 작성 과정에서 에러가 발생했습니다. \n ${error}`);
    }
  }

  function exportChange(e) {
    e.preventDefault();
    exportDocs();
  }
  
	useEffect(() => {
		const fetchData = async () => {
			try {
				const response = await axios.get(graphurl, config);
				const data = response.data;
				setData(data);
				setonLoad(false);
				} catch (error) {
					alert(`질병 확률 데이터 로딩 시 에러가 발생했습니다. \n ${error}`);
					setonLoad(true);
				}
		};
		fetchData();
	}, []);

  useEffect(() => {
    const fetchImage = async () => {
      try {
        const response_original = await axios.get(originalurl, config);
        const response_grad = await axios.get(gradcamurl + '?threshold=' + gradthres / 100, config);
        const data_original = response_original.data;
        const data_grad = response_grad.data;
        setImages(data_original)
        setGradImages(data_grad)
        setImageExists(true);
        setonLoad(false);
      } catch (error) {
        alert(`이미지 로딩 시 에러가 발생했습니다. \n ${error}`);
        setonLoad(true);
        setImageExists(false);
      }
    };
    fetchImage();
  }, [disease, gradthres]);

  useEffect(() => {
    const fetchPatient = async () => {
      try {
        const response = await axios.get(patienturl, config)
        setPatientInfo(response.data.info);
        setPatientLabel(response.data.labels);
        setonPatientLoad(false);
      } catch (error) {
        alert(`환자정보 로딩 시 에러가 발생했습니다. \n ${error}`);
        setonPatientLoad(true);
      }
    };
    fetchPatient();
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
    setPage(gradstate ? gradimages : images);
    }, [gradstate, images, gradimages]);

  const handleImageError = () => {
		setImageExists(false);
	  };

  const aboptions = {
		indexAxis: 'y',
		elements: {
		  bar: {
			borderWidth: 0.5,
		  },
		},
		aspectRatio: 10,
    maintainAspectRatio: false,
		responsive: true,
    animation: {
      duration: 0
    },
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
      x: {
        min:0,
        max:100,
        ticks: {
          // display: false,
        },
  
        grid: {
          drawBorder: false,
          display: false,
        },
      },
      y: {
        ticks: {
          display: false,
          beginAtZero: true,
        },
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
      animation: {
        duration: 0
      },
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
        x: {
          min: 0,
          max: 100,
          ticks: {
            // display: false,
          },
    
          grid: {
            drawBorder: false,
            display: false,
          },
        },
        y: {
          ticks: {
            display: false,
            beginAtZero: true,
          },
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
              <div className="text-wrapper">{!onPatientLoad ? `${patientLabel[0]}: ${patientInfo[0]} \u00A0 ${patientLabel[1]}: ${patientInfo[1]} \u00A0 
              ${patientLabel[2]}: ${patientInfo[2]} \u00A0 ${patientLabel[3]}: ${patientInfo[3]} \u00A0 ${patientLabel[4]}: ${patientInfo[4]}` : 'Loading...'}</div>
            </div>
              <div className="export">
                <Button variant="contained" onClick={exportChange} sx={{ color: 'white', backgroundColor: 'black', '&:hover': {
                backgroundColor: 'white', color: 'black'
                } }}>
                  Export
                </Button>
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
              <div className="text-wrapper-5">{onLoad ? '?' : `${abnormaldataset.datasets[0].data[0].x}%`}</div>
              <div className="text-wrapper-6"></div>
            </div>
          </div>
          <div className="category">
            <div className="text-wrapper-7">Category</div>
            <div className='slider'>
              <Slider
                defaultValue={50}
                step={null}
                valueLabelDisplay="auto"
                marks={marks}
                onChange={sliderChange}
                sx = {{ color: 'white' }}
              />
            </div>
            <div className='switch'>
              <Button variant="contained" onClick={showGrad} sx={{ color: 'white', backgroundColor: 'black', '&:hover': {
              backgroundColor: 'white', color: 'black'
              } }}>
                Inspect
              </Button>
            </div>
            <Link to="/results/abnormal">
            <div className="abnormal-button">
              <div className="overlap-group-2">
                <div className="abnormal-cat" style={styleList[0] ? whiteCircleStyle : blackCircleStyle}/>
                <div className="text-wrapper-9" style={styleList[0] ? blackTextStyle : whiteTextStyle}>Abnormal</div>
              </div>
            </div>
            </Link>
            <Link to="/results/meniscus">
            <div className="meniscus-button">
              <div className="overlap-group-2">
                <div className="meniscus-cat" style={styleList[2] ? whiteCircleStyle : blackCircleStyle}/>
                <div className="text-wrapper-10" style={styleList[2] ? blackTextStyle : whiteTextStyle}>Meniscus</div>
              </div>
            </div>
            </Link>
            <Link to="/results/acl">
            <div className="ACL-button">
              <div className="div-wrapper" style={styleList[1] ? whiteCircleStyle : blackCircleStyle}>
                <div className="text-wrapper-11" style={styleList[1] ? blackTextStyle : whiteTextStyle}>ACL</div>
              </div>
            </div>
            </Link>
          </div>
        </div>
        <div className="ACL-score">
          <div className="overlap-4">
            <div className="text-wrapper-12">ACL Score</div>
            <div className="overlap-5">
              <div className="rectangle-2">
                <Bar options={acloptions} data={acldataset} />
              </div>
              <div className="text-wrapper-13">{onLoad ? '?' : `${acldataset.datasets[0].data[0].x}%`}</div>
              <div className="text-wrapper-14"></div>
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
              <div className="text-wrapper-13">{onLoad ? '?' : `${meniscusdataset.datasets[0].data[0].x}%`}</div>
              <div className="text-wrapper-14"></div>
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
              <div className='overlap-group-3'>
                {onLoad ? (
                    <span>Loading...</span>
                  ) : imageExists ? (
                    page[0] ? (
                      <img
                        className="overlap-group-3-image"
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
                {gradstate ? 
                <div className="heatmap-bar">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-20">0</div>
                  <div className="text-wrapper-21">100</div>
                </div> :
                <div />}
            </div>
            <Link to={`/results/${disease}/axial`}>
            <div className="inspect-ax">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            </Link>
            {/* <div className="plus-ax">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div> */}
          </div>
        </div>
        <div className="coronal-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
                <div className='overlap-group-3'>
                {onLoad ? (
                    <span>Loading...</span>
                  ) : imageExists ? (
                    page[1] ? (
                      <img
                        className="overlap-group-3-image"
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
                {gradstate ? 
                <div className="heatmap-bar-2">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-20">0</div>
                  <div className="text-wrapper-21">100</div>
                </div> :
                <div />}
            </div>
            <Link to={`/results/${disease}/coronal`}>
            <div className="inspect-co">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            </Link>
            {/* <div className="plus-co">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div> */}
          </div>
        </div>
        <div className="sagittal-group">
          <div className="overlap-12">
            <div className="overlap-group-wrapper">
                <div className='overlap-group-3'>
                {onLoad ? (
                    <span>Loading...</span>
                  ) : imageExists ? (
                    page[2] ? (
                      <img
                        className="overlap-group-3-image"
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
                {gradstate ? 
                <div className="heatmap-bar-3">
                  <div className="rectangle-3" />
                  <div className="text-wrapper-20">0</div>
                  <div className="text-wrapper-21">100</div>
                </div> :
                <div />}
            </div>
            <Link to={`/results/${disease}/sagittal`}>
            <div className="inspect-sag">
              <div className="overlap-13">
                <div className="ellipse" />
                <div className="rectangle-4" />
              </div>
            </div>
            </Link>
            {/* <div className="plus-sag">
              <div className="overlap-14">
                <div className="rectangle-5" />
                <div className="rectangle-6" />
              </div>
            </div> */}
          </div>
        </div>
      </div>
    </div>
	)
}