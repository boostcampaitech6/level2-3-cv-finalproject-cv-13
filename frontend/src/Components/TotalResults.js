import React, {useState, useEffect} from 'react'
import axios from 'axios'
import './TotalResults.css'
import ImgAsset from '../public'
import {Link} from 'react-router-dom'
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


export default function TotalResults () {

	const [Data, setData] = useState([]);
	const [onLoad, setonLoad] = useState(true);

	const options = {
		indexAxis: 'y',
		elements: {
		  bar: {
			borderWidth: 0.5,
		  },
		},
		aspectRatio: 0.95,
		responsive: true,
		plugins: {
		  legend: {
			position: 'right',
		  },
		  title: {
			display: true,
			text: 'Results',
		  },
		},
	  };
	
	// Component가 처음 마운트 될 때 1번만 데이터를 가져옵니다
	useEffect(() => {
		const fetchData = async () => {
			try {
				const response = await axios.get("http://127.0.0.1:8000/totalresult");
				// console.log('response.data : ', response.data);
				const data = response.data;
				setData(data);
				setonLoad(false);
				} catch (error) {
					console.log(error);
					setonLoad(true);
				}
		};
		fetchData();
		// console.log('data : ', Data);
	}, []);

	let labels = [];
    if (Data.labels) {
        labels = Data.labels;
    }

	// console.log('labels : ', labels);
	// console.log('datasets : ', Data.datasets);

    const dataset = {
        labels,
        datasets: [
            {
                label: "%",
                data: Data.datasets,
                borderColor: "rgb(255, 99, 132)",
                backgroundColor: "rgba(255, 99, 132, 0.5)",
            },
        ],
    };

	// console.log('dataset[0] : ', dataset.datasets[0]);
	// console.log('onLoad : ', onLoad);
	let abnormalp, aclp, meniscusp = null;

	if (onLoad === true) {
		abnormalp = <span className='AbnormalPercent'>?% Abnormal</span>
		aclp = <span className='ACLPercent'>?% ACL</span>
		meniscusp = <span className='MeniscusPercent'>?% Meniscus</span>
	} else {
		abnormalp = <span className='AbnormalPercent'>{dataset.datasets[0].data[0].x}% Abnormal</span>
		aclp = <span className='ACLPercent'>{dataset.datasets[0].data[1].x}% ACL</span>
		meniscusp = <span className='MeniscusPercent'>{dataset.datasets[0].data[2].x}% Meniscus</span>
	}

	return (
		<div className='TotalResults_TotalResults'>
			<img className='background' src = {ImgAsset.background} />
			<div className='Overlay'>
				<div className='Buttons'>
					<Link to='/abnormalresultscoronal'>
						<div className='Abnormal'>
							<img className='Rectangle1' src = {ImgAsset.whiterectangle} />
							<span className='Abnormal_1'>Abnormal</span>
						</div>
					</Link>
					<Link to='/aclresultscoronal'>
						<div className='ACL'>
							<img className='Rectangle1_1' src = {ImgAsset.whiterectangle} />
							<span className='ACL_1'>ACL</span>
						</div>
					</Link>
					<Link to='/meniscusresultscoronal'>
						<div className='Meniscus'>
							<img className='Rectangle1_2' src = {ImgAsset.whiterectangle} />
							<span className='Meniscus_1'>Meniscus</span>
						</div>
					</Link>
					<div className='Total'>
						<img className='Rectangle1_3' src = {ImgAsset.blackrectangle} />
						<span className='Total_1'>Total</span>
					</div>
				</div>
				<img className='BubbleA' src = {ImgAsset.overlay4} />
				<img className='logo' src = {ImgAsset.logo} />
				<span className='Text2explain'>The Model thinks that the patient is...</span>
			</div>
			<div className='Contents'>
				<div className='PercentText'>
					<Link to='/abnormalresultscoronal'>
						{abnormalp}
					</Link>
					<Link to='/aclresultscoronal'>
						{aclp}
					</Link>
					<Link to='/meniscusresultscoronal'>
						{meniscusp}
					</Link>
				</div>
				<div className='Graph'>
					{<Bar options={options} data={dataset} />}
				</div>
			</div>
		</div>
	)
}