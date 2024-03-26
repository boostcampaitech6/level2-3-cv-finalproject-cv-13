import React from 'react'
import './InputTemplate.css'
import ImgAsset from '../public'
import axios from 'axios'
import { useHistory } from 'react-router-dom'
import {useState, useRef, useEffect} from 'react';
import {Button} from "@mui/material";

export default function FirstImpression (props) {

	const inputRefs = useRef({ axial: null, coronal: null, sagittal: null })
	const history = useHistory();
	const [loading, setLoading] = useState(false);
	let [ready, setReady] = useState([false, false, false]);
	const ip = localStorage.getItem('userIP') || props.ip;

	const saveImage = async (e, plane, idx) => {
	e.preventDefault();
	const files = e.target.files;
	const formData = new FormData();

	if (files) {
		for (let i = 0; i < files.length; i++) {
		formData.append('file', files[i]);
		}

		const currenturl = `/input/${plane}`
		const postOptions = {
			method: "POST",
			url: currenturl,
			body: formData,
			headers: {
				'IP': ip,
			},
		}

		try {
			setLoading(true);
            const response = await fetch(currenturl, postOptions);
			if (response.status == '400') {
				throw new TypeError('지원되지 않는 파일 형식입니다.');
			}
			else if (response.status == '500') {
				throw new Error('손상되었거나 지원되지 않는 DICOM 파일입니다.');
			}
			else if (response.status != '200') {
				throw new Error('파일 업로드 과정에서 에러가 발생했습니다.');
			}
			ready[idx] = true;
			setReady(ready);
        } catch (err) {
            alert(`파일 업로드에 실패했습니다. \n ${err}`);
        } finally {
			setLoading(false);
		}
	}
	};

	function changeScreen(e) {
		e.preventDefault();
		history.push("/loading");
	}

	const sampleData = async (e, disease) => {
		e.preventDefault();
		const sampleurl = `/input/sample?disease=${disease}`
		const config = {
			headers: {
			  'IP': ip,
			}
		};
		try {
			const response = await axios.get(sampleurl, config);
			setReady([true, true, true]);
			alert("샘플 파일 업로드에 성공했습니다. \nInference 버튼을 눌러 주세요");
		} catch (err) {
			alert(`샘플 파일 업로드에 실패했습니다. \n ${err}`);
		}
	}

	function alertUpload(e) {
		e.preventDefault();
		alert("DICOM 파일을 먼저 업로드해 주세요")
	}

	return (
	<div className="first-impression">
      <div className="div">
        <div className="image-holder-group">
          <div className="axial-complete">{ready[0] ? 'Complete!' : ''}</div>
          <div className="coronal-complete">{ready[1] ? 'Complete!' : ''}</div>
          <div className="sagittal-complete">{ready[2] ? 'Complete!' : ''}</div>
          <div className="axial-text">Upload Axial DICOM here...</div>
          <div className="coronal-text">Upload Coronal DICOM here...</div>
          <div className="sagittal-text">Upload Sagittal DICOM here...</div>
          <img className="axial-image" alt="Axial image" src= {ImgAsset.FirstImpression_AxialImage} />
          <img className="coronal-image" alt="Coronal image" src={ImgAsset.FirstImpression_AxialImage} />
          <img className="sagittal-image" alt="Sagittal image" src={ImgAsset.FirstImpression_AxialImage} />
          <div className="axialbutton">
				<input
					type="file"
					multiple={false}
					accept=".dcm"
					onChange={(e) => saveImage(e, 'axial', 0)}
					ref={refParam => inputRefs.current.axial = refParam}
					style={{ display: "none" }}
				/>
				<Button variant="contained" onClick={() => inputRefs.current.axial.click()} sx={{ color: 'white', backgroundColor: 'black', '&:hover': {
              backgroundColor: 'white', color: 'black'
              } }}>
					{loading ? 'Loading...' : 'Upload'}
				</Button> 
		  </div>
          <div className="coronalbutton">
		  		<input
					type="file"
					multiple={false}
					accept=".dcm"
					onChange={(e) => saveImage(e, 'coronal', 1)}
					ref={refParam => inputRefs.current.coronal = refParam}
					style={{ display: "none" }}
				/>
				<Button variant="contained" onClick={() => inputRefs.current.coronal.click()} sx={{ color: 'white', backgroundColor: 'black', '&:hover': {
              backgroundColor: 'white', color: 'black'
              } }}>
					{loading ? 'Loading...' : 'Upload'}
				</Button>
		  </div>
          <div className="sagittalbutton">
		  		<input
					type="file"
					multiple={false}
					accept=".dcm"
					onChange={(e) => saveImage(e, 'sagittal', 2)}
					ref={refParam => inputRefs.current.sagittal = refParam}
					style={{ display: "none" }}
				/>
				<Button variant="contained" onClick={() => inputRefs.current.sagittal.click()} sx={{ color: 'white', backgroundColor: 'black', '&:hover': {
              backgroundColor: 'white', color: 'black'
              } }}>
					{loading ? 'Loading...' : 'Upload'}
				</Button>
		  </div>
		    <div className='InferenceButton'>
				{ready.every((x) => x === true) ? 
				<Button variant="contained"  onClick={changeScreen}  sx={{ color: 'white', backgroundColor: 'black', '&:hover': {
					backgroundColor: 'white', color: 'black'
					} }}>
					Inference
				</Button>
				 :
				 <Button variant="contained" onClick={alertUpload}  sx={{ color: 'white', backgroundColor: 'black', '&:hover': {
					backgroundColor: 'white', color: 'black'
					} }}>
					Inference
				</Button>}
			</div>
			<div className='sample-acl'>
				<Button variant="text"  onClick={(e) => sampleData(e, 'acl')}  sx={{ color: 'white', fontFamily: "Alata, Helvetica", fontSize: '15px'}}>
					Upload Sample ACL file instead
				</Button>
			</div>
			<div className='sample-meniscus'>
				<Button variant="text"  onClick={(e) => sampleData(e, 'meniscus')}  sx={{ color: 'white', fontFamily: "Alata, Helvetica", fontSize: '15px'}}>
					Upload Sample Meniscus file instead
				</Button>
			</div>
        </div>
        <img className="logo" alt="Logo" src={ImgAsset.ResultsCodeACL_logo} />
      </div>
    </div>
	)
}