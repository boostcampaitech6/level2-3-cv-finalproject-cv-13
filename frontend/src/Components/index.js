import React from 'react'
import {Link} from 'react-router-dom'
export default function HomePage () {
    return (
	<div>
		<Link to='/ResultsCodeACL'><div>ResultsCodeACL</div></Link>
		<Link to='/ResultsCodeAbnormal'><div>ResultsCodeAbnormal</div></Link>
		<Link to='/ResultsCodeMeniscus'><div>ResultsCodeMeniscus</div></Link>
		<Link to='/AxialResults'><div>AxialResults</div></Link>
		<Link to='/CoronalResults'><div>CoronalResults</div></Link>
		<Link to='/SagittalResults'><div>SagittalResults</div></Link>
		<Link to='/FirstImpression'><div>FirstImpression</div></Link>
		<Link to='/Loading'><div>Loading</div></Link>
	</div>
	)
}