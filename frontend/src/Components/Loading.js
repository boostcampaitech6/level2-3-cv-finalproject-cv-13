import React, {useEffect} from 'react'
import axios from 'axios'
import './Loading.css'
import ImgAsset from '../public'
import { useHistory } from 'react-router-dom';

export default function Loading (props) {

    const history = useHistory();
	const ip = localStorage.getItem('userIP') || props.ip;
	
	useEffect(() => {
		const awaitInference = async () => {
			try {
				const url = "/inference";
				const config = {
					headers: {
					  'IP': ip,
					}
				};
				const response = await axios.get(url, config);
				history.push("/results/abnormal");
				} catch (error) {
					console.log(error);
					alert(`예측 도중에 에러가 발생하였습니다. \n ${error}`);
					history.push("/")
				}
		};
		awaitInference();
	}, []);

	return (
	<div className="loading">
      <div className="div">
        <div className="loading-group">
          <div className="text">Loading...</div>
          <img className="img" alt="Loading" src={ImgAsset.Loading_loading} />
        </div>
        <img className="logo" alt="Logo" src={ImgAsset.ResultsCodeACL_logo} />
      </div>
    </div>
	)
}