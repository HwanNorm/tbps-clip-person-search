import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Search from './pages/Search'
import Manage from './pages/Manage'
import './index.css'

function Nav() {
  return (
    <nav className="bg-gray-900 text-white px-6 py-3 flex gap-6 items-center">
      <span className="font-bold text-lg mr-4">PersonSearch</span>
      <NavLink
        to="/"
        end
        className={({ isActive }) =>
          isActive ? 'text-blue-400 font-semibold' : 'text-gray-300 hover:text-white'
        }
      >
        Search
      </NavLink>
      <NavLink
        to="/manage"
        className={({ isActive }) =>
          isActive ? 'text-blue-400 font-semibold' : 'text-gray-300 hover:text-white'
        }
      >
        Manage Datasets
      </NavLink>
    </nav>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        <Nav />
        <Routes>
          <Route path="/" element={<Search />} />
          <Route path="/manage" element={<Manage />} />
        </Routes>
      </div>
    </BrowserRouter>
  </React.StrictMode>,
)
