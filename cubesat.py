# -*- coding: utf-8 -*-

import os
import yaml
import glob
import math
import ephem
import pickle
import datetime
import numpy as np
import pandas as pd
# import healpy as hp


from pyorbital.orbital import Orbital
from pyorbital.tlefile import Tle
from pyorbital import tlefile

from skyfield.api import EarthSatellite
from skyfield.api import load

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as mdates

# COLUMN_KEYWORDS = ['utc_time','longitude_deg','latitude_deg','altitude_km','sun_elevation_deg','sat_elevation_deg','sat_azimuth_deg','cutoff_rigidity_GV',
# 	'maxi_rbm_rate_cps','hv_allowed_flag']

COLUMN_KEYWORDS = ['utc_time','longitude_deg','latitude_deg','altitude_km','dcm_x','sat_elevation_deg','sat_azimuth_deg','cutoff_rigidity_GV',
	'maxi_rbm_rate_cps','hv_allowed_flag']

class CubeSat():
	def __init__(self,setup_yamlfile,data_yamlfile,start_date_utc,end_date_utc,timebin_minute,fountname_base):

		print("init")

		self.param = yaml.load(open(setup_yamlfile),Loader=yaml.SafeLoader)

		self.orbital_tle = Tle(self.param['tle_name'],self.param['tle_file'])
		# self.orbital_tle = tlefile.read(self.param['tle_file'])
		self.orbital_orbit = Orbital(self.param['tle_name'], line1=self.orbital_tle.line1, line2=self.orbital_tle.line2)
		self.ts = load.timescale()
		self.skyfield_satellite = EarthSatellite(self.orbital_tle.line1, self.orbital_tle.line2, 'ISS', self.ts)

		self.ephem_sun = ephem.Sun()
		self.ephem_sat = ephem.Observer()

		self.df_column_keywords = ["Time", "Longitude", "Latitude", "Altitude"]
		self.df = None
		self.utc_time_init = start_date_utc
		self.utc_time_end = end_date_utc
		self.time_bin_min = timebin_minute
		# self.dcm_init = None
		# self.q_init = np.array([0, 0, 0, 0])
		# self.dcm_init = np.array([[1, 0, 0],
		#         					[0, 1, 0],
		#         					[0, 0, 1]])

		self.target_df = pd.read_csv(self.param['obs_target_list'],skiprows=[0,1],sep=',')
		self.gs_df = pd.read_csv(self.param['ground_station_list'],sep=',')

		self.longitude_deg = None # 経度
		self.latitude_deg = None # 緯度
		self.altitude_km = None
		self.sun_elevation_deg = None
		self.sat_elevation_deg = None
		self.sat_azimuth_deg = None

		self.maxi_rbm_map_h = None
		self.maxi_rbm_map_z = None
		self.maxi_rbm_map_index_h = None
		self.maxi_rbm_map_index_z = None

		self.track_dict = {}
		for keyword in self.df_column_keywords:
			self.track_dict[keyword] = []

		self.DEG2RAD = math.pi / 180
		self.ARCSEC2RAD = self.DEG2RAD /36000
		self.MJD_J2000 = 51544.5 # The value depends on the context, this is a commonly used value
		self.EARTH_RADIUS = 6371 # in kilometers
		self.MOON_RADIUS = 1738 # in kilometers
		self.EPS = 1e-9


## Create dataframe and simulate satellite position vs time

	def simulate_orbit(self):
		"""
		Input : self
		Output : time is propageted to the time_end and full orbit is simulated
					full dataframe is returned
		"""
		last_row = self.df.iloc[-1]
		while pd.to_datetime(last_row['Time']) < self.str2datetime(self.utc_time_end):
			self.update_df()
			last_row = self.df.iloc[-1]
			print(pd.to_datetime(last_row['Time']))

	def setup_df(self):
		"""
		set up initial dataframe which has the following keywords
		["Time" : string
		"Longitude""Altitude" float
		"""
		self.df = pd.DataFrame(columns=self.df_column_keywords)
		longitude, latitude, altitude = self.get_position(self.str2datetime(self.utc_time_init))
		# q_1d = self.q_init
		# row = [self.utc_time_init, longitude, latitude, altitude] + q_1d.tolist()
		row = [self.utc_time_init, longitude, latitude, altitude]
		self.df = self.df.append(pd.DataFrame([row], columns=self.df_column_keywords), ignore_index=True)

	def update_df(self):
		"""
		with adding the time delta to the time in the last row in the dataframe, append a new position with the new time to the dataframe
		"""
		last_row = self.df.iloc[-1]
		new_time = pd.to_datetime(last_row['Time']) + pd.Timedelta(minutes=self.time_bin_min)
		new_longitude, new_latitude, new_altitude = self.get_position(new_time)
		row = [self.datetime2str(new_time), new_longitude, new_latitude, new_altitude]
		self.df = self.df.append(pd.DataFrame([row], columns=self.df_column_keywords), ignore_index=True)


	def add_position_TEME(self):
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])
			p, v = self.orbital_orbit.get_position(datetime_value, False)
			self.df.loc[index, "xTEME"] = p[0]
			self.df.loc[index, "yTEME"] = p[1]
			self.df.loc[index, "zTEME"] = p[2]

	def add_position_J2000(self):
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])
			x_pos, y_pos, z_pos = self.get_position_J2000(datetime_value)
			self.df.loc[index, 'xJ2000'] = x_pos
			self.df.loc[index, 'yJ2000'] = y_pos
			self.df.loc[index, 'zJ2000'] = z_pos

	def get_position_J2000(self, datetime_value):
		# print(datetime_value)
		year = datetime_value.year
		month = datetime_value.month
		day = datetime_value.day
		hour = datetime_value.hour
		minute = datetime_value.minute
		second = datetime_value.second
		t = self.ts.utc(year, month, day, hour, minute, second)
		geocentric = self.skyfield_satellite.at(t)
		# print(geocentric.position.km)
		# print(geocentric.position.km[0])
		x_pos = geocentric.position.km[0]
		y_pos = geocentric.position.km[1]
		z_pos = geocentric.position.km[2]
		return x_pos, y_pos, z_pos

	def add_sun_moon_position(self):
		for index, row in self.df.iterrows():
				datetime_value = self.str2datetime(row["Time"])
				mjd_value = self.datetime2mjd(datetime_value)
				moonVect = self.atMoon(mjd_value)[0] / np.linalg.norm(self.atMoon(mjd_value)[0])
				sunVect = self.atSun(mjd_value)/ np.linalg.norm(self.atSun(mjd_value))

				self.df.loc[index, 'xSunVect'] = sunVect[0]
				self.df.loc[index, 'ySunVect'] = sunVect[1]
				self.df.loc[index, 'zSunVect'] = sunVect[2]

				self.df.loc[index, 'xMoonVect'] = moonVect[0]
				self.df.loc[index, 'yMoonVect'] = moonVect[1]
				self.df.loc[index, 'zMoonVect'] = moonVect[2]

	def add_target_vis(self):
		for index, row in self.target_df.iterrows():
			# print(row)
			# print(index)
			target_ra = row["RA (radians)"]
			target_dec = row["DEC (radians)"]
			xVect = self.get_direction_vector(target_ra, target_dec)

			for index2, row2 in self.df.iterrows():
				datetime_value = self.str2datetime(row2["Time"])
				mjd_value = self.datetime2mjd(datetime_value)
				x_pos, y_pos, z_pos = self.get_position_J2000(datetime_value)
				satVect = np.array([x_pos, y_pos, z_pos])
				moonVect = self.atMoon(mjd_value)[0] / np.linalg.norm(self.atMoon(mjd_value)[0])
				sunVect = self.atSun(mjd_value)/ np.linalg.norm(self.atSun(mjd_value))
				earthVect = -satVect
				flag, el = self.at_earth_occult(satVect, xVect, sunVect)
				self.df.loc[index2, 'visFlag_'+row["Name"]] = flag
				self.df.loc[index2, 'visEl_'+row["Name"]] = el

				self.df.loc[index2, 'angDistMoon_'+row["Name"]] = self.ang_distance(xVect, moonVect)
				self.df.loc[index2, 'angDistSun_'+row["Name"]] = self.ang_distance(xVect, sunVect)

				self.df.loc[index2, 'xTargVec_'+row["Name"]] = xVect[0]
				self.df.loc[index2, 'yTargVec_'+row["Name"]] = xVect[1]
				self.df.loc[index2, 'zTargVec_'+row["Name"]] = xVect[2]

	def setup_target_df(self):
		self.target_df = pd.read_csv(self.param['obs_target_list'],skiprows=[0,1],sep=',')
		self.target_df['RA (radians)'] = self.target_df['RA (J2000)'].apply(self.angtime2radians)
		self.target_df['DEC (radians)'] = self.target_df['DEC (J2000)'].apply(lambda x: self.angtime2radians(x, ra=False))

	


	def at_earth_occult(self, sat_vect, x_vect, sun_vect):
		sat_v, earth_vect, x_v = None, None, None
		earth_vect = -sat_vect
		sat_distance = self.norm(earth_vect)
		earth_size = np.arcsin(self.EARTH_RADIUS / sat_distance)
		x_dist = self.ang_distance(x_vect, earth_vect)
		el = x_dist - earth_size

		flag = 0
		if el <= -self.EPS:
			rm = self.set_rot_mat_zx(sun_vect, sat_vect)
			sat_v = self.rot_vect(rm, sat_vect)
			x_v = self.rot_vect(rm, x_vect)
			dot = self.scal_prod(earth_vect, x_vect)
			z_cross = sat_v[2] + x_v[2] * (dot
				- np.sqrt(self.EARTH_RADIUS * self.EARTH_RADIUS
				- sat_distance * sat_distance + dot * dot))
			if z_cross < 0.:
				flag = 1  # Dark Earth
			else:
				flag = 2  # Bright Earth
		return flag, el


	def get_direction_vector(self, lat_rad, lon_rad):

		# XYZ座標を計算
		x = np.cos(lat_rad) * np.cos(lon_rad)
		y = np.cos(lat_rad) * np.sin(lon_rad)
		z = np.sin(lat_rad)

		# 方向ベクトルを返す
		return np.array([x, y, z])


	def add_observer_vis(self):
		for index, row in self.gs_df.iterrows():
			for index2, row2 in self.df.iterrows():
				datetime_value = self.str2datetime(row2["Time"])
				st_azi_deg, st_ele_deg = self.orbital_orbit.get_observer_look(datetime_value, row['long'], row['lat'], row['alt'])
				self.df.loc[index2, 'gsElev_'+row["Name"]] = st_ele_deg
				self.df.loc[index2, 'gsAzim_'+row["Name"]] = st_azi_deg
				self.df.loc[index2, 'gsFlag_'+row["Name"]] = st_ele_deg>self.param['sight_elevation']

	def plot_observer_vis(self):
		fig = plt.figure(figsize=(12,8))
		plt.clf()
		ax = plt.axes(projection=ccrs.PlateCarree())

		# Loop for dataframe, time series
		for index2, row2 in self.df.iterrows():
			# total index flags, 0 means no communication with any ground station, 1 means communication with a ground station
			obs_vis_ind = 0
			# Loop for ground stations, location series
			for index, row in self.gs_df.iterrows():

				if self.df.loc[index2, 'gsFlag_'+row["Name"]] == 1:
					obs_vis_ind = 1
					# a ground station has been found, plot the location as a communication point
					break
				else:
					obs_vis_ind = 0

			if obs_vis_ind == 1:
				ax.scatter(self.df.loc[index2, 'Longitude'], self.df.loc[index2, 'Latitude'],
					transform=ccrs.PlateCarree(), marker='x', s=20, c="r")
			else:
				ax.scatter(self.df.loc[index2, 'Longitude'], self.df.loc[index2, 'Latitude'],
					transform=ccrs.PlateCarree(), marker='x', s=20, c="b")

		# add text label on each location
		for i in range(len(self.df)):
			ax.text(self.df['Longitude'][i], self.df['Latitude'][i], self.df['Time'][i], fontsize=8)

		# plot ground station location, adustment of "+2" for avoiding overlap
		for index_gs, row_gs in self.gs_df.iterrows():
			ax.scatter(row_gs['long'], row_gs['lat'],
				transform=ccrs.PlateCarree(), marker='D', s=30, c="m")
			ax.text(row_gs['long']+2, row_gs['lat']+2, row_gs['Name'], transform=ccrs.PlateCarree(), fontsize=10, c="m")

		ax.set_global()
		ax.coastlines(linewidth=0.5)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg)')
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])


	def setup_maxi_rbm_index(self):

		maxi_rbm_image_list = []
		with open(self.param['maxi_rbm_pic_directory'], 'rb') as maxi:
			maxi_rbm_image_lst = pickle.load(maxi)

		self.maxi_rbm_map_h = maxi_rbm_image_lst[self.param['maxi_rbm_selected_basename_h']]
		self.maxi_rbm_map_h = np.nan_to_num(self.maxi_rbm_map_h)
		maxi_rbm_map_index_bool_h = self.maxi_rbm_map_h > self.param['hv_allowed_maxi_rbm_threshold']
		self.maxi_rbm_map_index_h = maxi_rbm_map_index_bool_h.astype(np.int)

		self.maxi_rbm_map_z = maxi_rbm_image_lst[self.param['maxi_rbm_selected_basename_z']]
		self.maxi_rbm_map_z = np.nan_to_num(self.maxi_rbm_map_z)
		maxi_rbm_map_index_bool_z = self.maxi_rbm_map_z > self.param['hv_allowed_maxi_rbm_threshold']
		self.maxi_rbm_map_index_z = maxi_rbm_map_index_bool_z.astype(np.int)

	def add_maxi_rbm_index(self):
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])

			theta = np.linspace(-180, 180, 1441)
			phi = np.linspace(0,180, 481)

			long=self.orbital_orbit.get_lonlatalt(datetime_value)[0]
			lat=self.orbital_orbit.get_lonlatalt(datetime_value)[1]

			index_theta = np.digitize(lat, theta)
			index_phi = np.digitize(long, phi)
			maxi_rbm_h = self.maxi_rbm_map_h.T[index_phi][index_theta]
			maxi_rbm_z = self.maxi_rbm_map_z.T[index_phi][index_theta]

			maxi_rbm_index_h = self.maxi_rbm_map_index_h.T[index_phi][index_theta]
			maxi_rbm_index_z = self.maxi_rbm_map_index_z.T[index_phi][index_theta]

			self.df.loc[index, 'hMaxi'] = maxi_rbm_h
			self.df.loc[index, 'zMaxi'] = maxi_rbm_z
			self.df.loc[index, 'hMaxiInd'] = maxi_rbm_index_h
			self.df.loc[index, 'zMaxiInd'] = maxi_rbm_index_z

	def plot_maxi_rbm_flag(self):
		fig = plt.figure(figsize=(12,8))
		plt.clf()
		ax = plt.axes(projection=ccrs.PlateCarree())
		im = ax.imshow(self.maxi_rbm_map_index_h.T,
			origin="lower",extent=[-180, 180, -90, 90],
			transform=ccrs.PlateCarree(), cmap='Greys',	alpha=0.4)
		cbar=plt.colorbar(im, shrink=0.5, pad=0.1)
		cbar.set_label("RBM counts s$^{-1}$ [%s]" % self.param['maxi_rbm_selected_basename_h'])

		ax.scatter(self.df['Longitude'], self.df['Latitude'],transform=ccrs.PlateCarree(), marker='x', s=20, c="r")
		for i in range(len(self.df)):
			ax.text(self.df['Longitude'][i], self.df['Latitude'][i], self.df['Time'][i], fontsize=8)

		ax.set_global()
		ax.coastlines(linewidth=0.5)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg), [{}], threshold={}'.format(self.param['maxi_rbm_selected_basename_h'], self.param['hv_allowed_maxi_rbm_threshold']))
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])

	def plot_maxi_rbm_map(self):
		fig = plt.figure(figsize=(12,8))
		plt.clf()
		ax = plt.axes(projection=ccrs.PlateCarree())
		# map rbm values
		im = ax.imshow(self.maxi_rbm_map_h.T,
			origin="lower",extent=[-180, 180, -90, 90],
			transform=ccrs.PlateCarree(), alpha=0.4, norm=LogNorm(), cmap='Greys')
		cbar=plt.colorbar(im, shrink=0.5, pad=0.1)
		cbar.set_label("RBM counts s$^{-1}$ [%s]" % self.param['maxi_rbm_selected_basename_h'])
		# contour rbm levels
		contours=ax.contour(np.where(self.maxi_rbm_map_h.T <= 0, np.nan, self.maxi_rbm_map_h.T), origin="lower"
				,levels=[0.1, 1, 10, 100, 1000, 10000],	norm=LogNorm(), extent=[-180, 180, -90, 90], cmap='Greys')
		ax.clabel(contours, inline=1, fontsize=8)
		# scatter satellite location
		ax.scatter(self.df['Longitude'], self.df['Latitude'],transform=ccrs.PlateCarree(), marker='x', s=20, c="r")
		for i in range(len(self.df)):
			ax.text(self.df['Longitude'][i], self.df['Latitude'][i], self.df['Time'][i], fontsize=8)
		ax.set_global()
		ax.coastlines(linewidth=0.5)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg), [{}], threshold={}'.format(self.param['maxi_rbm_selected_basename_h'], self.param['hv_allowed_maxi_rbm_threshold']))
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])

	def set_cutoff_rigidity(self):
		print("--set_cutoff_rigidity_map")
		print('file_cutoffrigidity: {}'.format(self.param['file_cutoffrigidity']))

		# file_cutoffrigidityは、yamlファイルからのインプット

		# 新しいインスタンス変数、COR = Cut Off Rigidity
		self.df_cor = pd.read_csv(self.param['file_cutoffrigidity'],
			skiprows=1,delim_whitespace=True,
			names=['latitude_deg','longitude_deg','cutoff_rigidity'],
			dtype={'latitude_deg':'float64','longitude_deg':'float64','cutoff_rigidity':'float64'})
		# print("df_cor")
		# print(self.df_cor)

		# 新しいインスタンス変数
		self.cutoff_rigidity_map, self.cormap_longitude_edges, self.cormap_latitude_edges = np.histogram2d([],[],
			bins=[self.param["cormap_lon_nbin"],self.param["cormap_lat_nbin"]],
			range=[[-180.,180.],[-90.,90.]])
		self.df_cor["longitude_index"] = np.digitize(self.df_cor['longitude_deg'], self.cormap_longitude_edges)
		self.df_cor["longitude_index"] = pd.to_numeric(self.df_cor["longitude_index"],downcast='signed')
		self.df_cor["latitude_index"] = np.digitize(self.df_cor['latitude_deg'], self.cormap_latitude_edges)
		self.df_cor["latitude_index"] = pd.to_numeric(self.df_cor["latitude_index"],downcast='signed')
		"""
		H: two dimentional matrix
		x: longitude
		y: latitude

		H[0][2]		...
		H[0][1]		H[1][1]		...
		H[0][0]		H[1][0]		...
		"""

		for index, row in self.df_cor.iterrows():
			i = int(row['longitude_index'])-1
			j = int(row['latitude_index'])-1
			self.cutoff_rigidity_map[i][j] = row['cutoff_rigidity']

		self.cutoff_rigidity_map_T = self.cutoff_rigidity_map.T
		self.cormap_longitude_centers = (self.cormap_longitude_edges[:-1] + self.cormap_longitude_edges[1:]) / 2.
		self.cormap_latitude_centers = (self.cormap_latitude_edges[:-1] + self.cormap_latitude_edges[1:]) / 2.

	def add_cutoff_rigidity(self):
		print("get_cutoff_rigidity")
		for index, row in self.df.iterrows():
			datetime_value = self.str2datetime(row["Time"])

			long = self.orbital_orbit.get_lonlatalt(datetime_value)[0]
			lat = self.orbital_orbit.get_lonlatalt(datetime_value)[1]

			i = int(np.digitize(long, self.cormap_longitude_edges)) - 1
			j = int(np.digitize(lat, self.cormap_latitude_edges)) - 1

			self.df.loc[index, 'COR'] = self.cutoff_rigidity_map[i][j]


	def plot_cutoff_rigidity_map(self,foutname_base='cormap'):
		print("plot_cutoff_rigidity_map")

		fig = plt.figure(figsize=(12,8))
		ax = plt.axes(projection=ccrs.PlateCarree())
		ax.stock_img()
		ax.coastlines(resolution='110m')

		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
			linewidth=1, color='gray', alpha=0.3, linestyle='--')
		ax.set_xlim(-180.0,180.0)
		ax.set_ylim(-90.0,90.0)
		ax.set_xlabel('Longitude (deg)')
		ax.set_ylabel('Latitude (deg)')
		ax.set_xticks([-180,-120,-60,0,60,120,180])
		ax.set_yticks([-90,-60,-30,0,30,60,90])
		X, Y = np.meshgrid(self.cormap_longitude_edges, self.cormap_latitude_edges)
		ax.pcolormesh(X, Y, self.cutoff_rigidity_map_T, alpha=0.5)
		cormap_contour = ax.contour(
			self.cormap_longitude_centers,
			self.cormap_latitude_centers,
			self.cutoff_rigidity_map_T,
			levels=10, colors='Black',
			transform=ccrs.PlateCarree())
		cormap_contour.clabel(fmt='%1.1f', fontsize=12)
		# plt.savefig("%s.pdf" % foutname_base,bbox_inches='tight')



	def set_hvoff_region(self):
		print("set_hvoff_region")
		# https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
		open(self.param["lookuptable_hvoff"])



	def datetime2mjd(self,time_datetime):
		julian_day = time_datetime.toordinal() + 1721425.5
		mjd = julian_day - 2400000.5
		return mjd

	def datetime2jd(self,time_datetime):
		julian_day = time_datetime.toordinal() + 1721425.5
		return julian_day

	def str2datetime(self, time_str):
		"""
		Assume time string such as '2021-08-08 23:59:00', and return as datetime object
		"""
		time_datetime = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
		return time_datetime

	def datetime2str(self, time_datetime):
		"""
		Assume time in datetime such as '2021-08-08 23:59:00', and return as string
		"""
		time_str = time_datetime.strftime('%Y-%m-%d %H:%M:%S')
		return time_str

	def get_position(self,utc_time):
		# print("get_position")
		"""
		return satellite position: longitude (deg), latitude (deg), and altitude (km)
		"""
		return self.orbital_orbit.get_lonlatalt(utc_time)



	def norm(self, vector):
		return np.linalg.norm(vector)

	def norm_vect(self, vector):
		norm = np.linalg.norm(vector)
		if norm == 0:
			return None
		return vector / norm

	def set_rot_mat_zx(self, z_axis, x_axis):
		y_axis = self.vect_prod(z_axis, x_axis)
		z = self.norm_vect(z_axis)
		y = self.norm_vect(y_axis)
		if z is None or y is None:
			return None
		x = self.vect_prod(y, z)

		rm = np.array([x, y, z])
		return rm

	def ang_distance(self, vector1, vector2):
		return np.arccos(np.dot(vector1, vector2) / (self.norm(vector1) * self.norm(vector2)))

	def vect_prod(self, vector1, vector2):
		return np.cross(vector1, vector2)

	def rot_vect(self, matrix, vector):
		return matrix.dot(vector)

	def scal_prod(self, vector1, vector2):
		return np.dot(vector1, vector2)

	def angtime2radians(self, time_str, ra=True):
		t = [int(i) for i in time_str.split(':')]

		if ra: # if the string is RA
			total_degrees = 15 * (t[0] + t[1]/60 + t[2]/3600)
		else: # if the string is DEC
			sign = -1 if time_str[0] == '-' else 1
			total_degrees = sign * (abs(t[0]) + t[1]/60 + t[2]/3600)
		return math.radians(total_degrees)


	def atPrecession(self, mjd0, x0, mjd):
		rm = np.zeros((3, 3))
		rm = self.atPrecessRM(mjd0, mjd)
		x = np.dot(rm, x0)
		return x

	def atPrecessRM(self, mjd0, mjd):
		RmAto2000 = np.zeros((3, 3))
		RmBto2000 = np.zeros((3, 3))
		Rm2000toB = np.zeros((3, 3))
		rm = np.zeros((3, 3))

		RmAto2000 = self.atPrecessRMJ2000(mjd0)
		RmBto2000 = self.atPrecessRMJ2000(mjd)
		Rm2000toB = np.transpose(RmBto2000)
		rm = np.dot(RmAto2000, Rm2000toB)

		return rm



	def atPrecessRMJ2000(self, mjd):
		t = (mjd - self.MJD_J2000) / 36525.0

		zeta = (2306.2181 + (0.30188 + 0.017998*t)*t)*t * self.ARCSEC2RAD
		z = (2306.2181 + (1.09468 + 0.018203*t)*t)*t * self.ARCSEC2RAD
		theta = (2004.3109 - (0.42665 + 0.041833*t)*t)*t * self.ARCSEC2RAD

		cos_zeta = math.cos(zeta)
		sin_zeta = math.sin(zeta)

		cos_z = math.cos(z)
		sin_z = math.sin(z)

		cos_theta = math.cos(theta)
		sin_theta = math.sin(theta)

		rm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
		rm[0][0] = cos_zeta*cos_theta*cos_z - sin_zeta*sin_z
		rm[1][0] = -sin_zeta*cos_theta*cos_z - cos_zeta*sin_z
		rm[2][0] = -sin_theta*cos_z
		rm[0][1] = cos_zeta*cos_theta*sin_z + sin_zeta*cos_z
		rm[1][1] = -sin_zeta*cos_theta*sin_z + cos_zeta*cos_z
		rm[2][1] = -sin_theta*sin_z
		rm[0][2] = cos_zeta*sin_theta
		rm[1][2] = -sin_zeta*sin_theta
		rm[2][2] = cos_theta

		return rm

	def atSun(self, mjd):
		MJD_B1950 = 33281.923
		DEG2RAD = np.pi / 180.0

		rm1950_2000 = self.atPrecessRMJ2000(MJD_B1950)

		t = mjd - 4.5e4
		m = ( np.fmod(t * .985600267, 360.0) + 27.26464 ) * DEG2RAD
		sin_2m = np.sin(2*m)

		l = ( np.fmod(t * .985609104, 360.0) - 50.55138
			  + np.sin(m) * 1.91553 + sin_2m * .0201 ) * DEG2RAD
		sin_l = np.sin(l)

		r = 1.00014 - np.cos(m) * .01672 - sin_2m * 1.4e-4

		x = np.array([r * np.cos(l), r * .91744 * sin_l, r * .39788 * sin_l])

		pos = np.dot(rm1950_2000, x)
		return pos

	def atMoon(self, mjd):  # input: time in MJD
		ta, a, b, c, d, e, g, j, l, m, n, v, w = self.moonag(mjd)
		mx, my, mz = self.moonth(ta, a, b, c, d, e, g, j, l, m, n, v, w)

		r_xy = math.sqrt(my - mx * mx)
		sin_delta = mz / r_xy
		cos_delta = math.sqrt(1. - sin_delta * sin_delta)
		sin_c = math.sin(c)
		cos_c = math.cos(c)

		# R.A. of moon = mean longitude (c) + delta
		distan = self.EARTH_RADIUS * math.sqrt(my)
		x_tod = [0, 0, 0]
		x_tod[0] = self.EARTH_RADIUS * r_xy * (cos_delta * cos_c - sin_delta * sin_c)
		x_tod[1] = self.EARTH_RADIUS * r_xy * (sin_delta * cos_c + cos_delta * sin_c)
		x_tod[2] = self.EARTH_RADIUS * mx

		size = math.atan(self.MOON_RADIUS / distan)
		phase = d % (np.pi * 2)
		pos = self.atPrecession(mjd, x_tod, self.MJD_J2000)

		return pos, size, phase, distan  # return as a tuple

	def moonag(self, mjd):
		DEG2RAD = np.pi / 180.0
		ta = (mjd - 15019.5) / 36525.
		tb = ta * ta

		a = DEG2RAD*(ta * 4.77e5 +296.1044608 + ta * 198.849108 + tb * .009192)
		b = DEG2RAD*(ta * 483120. + 11.250889 + ta * 82.02515 - tb * .003211)
		c = DEG2RAD*(ta * 480960. +270.434164 + ta * 307.883142 - tb * .001133)
		d = DEG2RAD*(ta * 444960 + 350.737486 + ta * 307.114217 - tb * .001436)
		e = DEG2RAD*(ta * 35640 + 98.998753 + ta * 359.372886)
		g = DEG2RAD*(ta * 35999.04975 + 358.475833 - tb * 1.5e-4)
		j = DEG2RAD*(ta * 2880 + 225.444651 + ta * 154.906654)
		l = DEG2RAD*(ta * 36000.76892 + 279.696678 + tb * 3.03e-4)
		m = DEG2RAD*(ta * 19080 + 319.529425 + ta * 59.8585 + tb * 1.81e-4)
		n = DEG2RAD*(259.183275 - ta * 1800 - ta * 134.142008 + tb * .002078)
		v = DEG2RAD*(ta * 58320 + 212.603219 + ta * 197.803875 + tb * .001286)
		w = DEG2RAD*(ta * 58320 + 342.767053 + ta * 199.211911 * 3.1e-4 * tb)

		return ta, a, b, c, d, e, g, j, l, m, n, v, w

	def moonth(self, ta, a, b, c, d, e, g, j, l, m, n, v, w):
		# MOON THETA
		mx = math.sin(a + b - d * 4.0) * -0.00101
		mx -= math.sin(a - b - d * 4.0 - n) * 0.00102
		mx -= ta * 0.00103 * math.sin(a - b - n)
		mx -= math.sin(a - g - b - d * 2.0 - n) * 0.00107
		mx -= math.sin(a * 2.0 - b - d * 4.0 - n) * 0.00121
		mx += math.sin(a * 3.0 + b + n) * 0.0013
		mx -= math.sin(a + b - n) * 0.00131
		mx += math.sin(a + b - d + n) * 0.00136
		mx -= math.sin(g + b) * 0.00145
		mx -= math.sin(a + g - b - d * 2.0) * 0.00149
		mx += math.sin(g - b + d - n) * 0.00157
		mx -= math.sin(g - b) * 0.00159
		mx += math.sin(a - g + b - d * 2.0 + n) * 0.00184
		mx -= math.sin(b - d * 2.0 - n) * 0.00194
		mx -= math.sin(g - b + d * 2.0 - n) * 0.00196
		mx += math.sin(b - d) * 0.002
		mx -= math.sin(a + g - b) * 0.00205
		mx += math.sin(a - g - b) * 0.00235
		mx += math.sin(a - b * 3 - n) * 0.00246
		mx -= math.sin(a * 2 + b - d * 2.0) * 0.00262
		mx -= math.sin(a + g + b - d * 2.0) * 0.00283
		mx -= math.sin(g - b - d * 2.0 - n) * 0.00339
		mx += math.sin(a - b + n) * 0.00345
		mx -= math.sin(g - b + d * 2.0) * 0.00347
		mx -= math.sin(b + d + n) * 0.00383
		mx -= math.sin(a + g + b + n) * 0.00411
		mx -= math.sin(a * 2 - b - d * 2.0 - n) * 0.00442
		mx += math.sin(a - b + d * 2.0) * 0.00449
		mx -= math.sin(b * 3 - d * 2.0 + n) * 0.00456
		mx += math.sin(a + b + d * 2.0 + n) * 0.00466
		mx += math.sin(a * 2 - b) * 0.0049
		mx += math.sin(a * 2 + b) * 0.00561
		mx += math.sin(a - g + b + n) * 0.00564
		mx -= math.sin(a + g - b - n) * 0.00638
		mx -= math.sin(a + g - b - d * 2.0 - n) * 0.00713
		mx -= math.sin(g + b - d * 2.0) * 0.00929
		mx -= math.sin(a * 2 - b - n) * 0.00947
		mx += math.sin(a - g - b - n) * 0.00965
		mx += math.sin(b + d * 2.0) * 0.0097
		mx += math.sin(b - d + n) * 0.01064
		mx -= ta * 0.0125 * math.sin(b + n)
		mx -= math.sin(g + b - d * 2.0 + n) * 0.01434
		mx -= math.sin(a + g + b - d * 2.0 + n) * 0.01652
		mx -= math.sin(a * 2 + b - d * 2.0 + n) * 0.01868
		mx += math.sin(a * 2 + b + n) * 0.027
		mx -= math.sin(a - b - d * 2.0) * 0.02994
		mx -= math.sin(g + b + n) * 0.03759
		mx -= math.sin(g - b - n) * 0.03982
		mx += math.sin(b + d * 2.0 + n) * 0.04732
		mx -= math.sin(b - n) * 0.04771
		mx -= math.sin(a + b - d * 2.0) * 0.06505
		mx += math.sin(a + b) * 0.13622
		mx -= math.sin(a - b - d * 2.0 - n) * 0.14511
		mx -= math.sin(b - d * 2.0) * 0.18354
		mx -= math.sin(b - d * 2.0 + n) * 0.20017
		mx -= math.sin(a + b - d * 2.0 + n) * 0.38899
		mx += math.sin(a - b) * 0.40248
		mx += math.sin(a + b + n) * 0.65973
		mx += math.sin(a - b - n) * 1.96763
		mx += math.sin(b) * 4.95372
		mx += math.sin(b + n) * 23.89684

		# MOON RHO
		my = math.cos(a * 2 + g) * 0.05491
		my += math.cos(a + d) * 0.0629
		my -= math.cos(d * 4) * 0.06444
		my -= math.cos(a * 2 - g) * 0.06652
		my -= math.cos(g - d * 4) * 0.07369
		my += math.cos(a - d * 3) * 0.08119
		my -= math.cos(a + d * 4) * 0.09261
		my += math.cos(a - b * 2 + d * 2) * 0.10177
		my += math.cos(a + g + d * 2) * 0.10225
		my -= math.cos(a + g * 2 - d * 2) * 0.10243
		my -= math.cos(b * 2) * 0.12291
		my -= math.cos(a * 2 - b * 2) * 0.12291
		my -= math.cos(a + g - d * 4) * 0.12428
		my -= math.cos(a * 3) * 0.14986
		my -= math.cos(a - g + d * 2) * 0.1607
		my -= math.cos(a - d) * 0.16949
		my += math.cos(a + b * 2 - d * 2) * 0.17697
		my -= math.cos(a * 2 - d * 4) * 0.18815
		my -= math.cos(g * 2 - d * 2) * 0.19482
		my += math.cos(b * 2 - d * 2) * 0.22383
		my += math.cos(a * 3 - d * 2) * 0.22594
		my += math.cos(a * 2 + g - d * 2) * 0.24454
		my -= math.cos(g + d) * 0.31717
		my -= math.cos(a - d * 4) * 0.36333
		my += math.cos(a - g - d * 2) * 0.47999
		my += math.cos(g + d * 2) * 0.63844
		my += math.cos(g) * 0.8617
		my += math.cos(a - b * 2) * 1.50534
		my -= math.cos(a + d * 2) * 1.67417
		my += math.cos(a + g) * 1.99463
		my += math.cos(d) * 2.07579
		my -= math.cos(a - g) * 2.455
		my -= math.cos(a + g - d * 2) * 2.74067
		my -= math.cos(g - d * 2) * 3.83002
		my -= math.cos(a * 2) * 5.37817
		my += math.cos(a * 2 - d * 2) * 6.60763
		my -= math.cos(d * 2) * 53.97626
		my -= math.cos(a - d * 2) * 68.62152
		my -= math.cos(a) * 395.13669
		my += 3649.33705

		# MOON PHI
		mz = math.sin(a - g - b * 2 - n * 2) * -0.001
		mz -= math.sin(a + g - d * 4) * 0.001
		mz += math.sin(a * 2 - g) * 0.001
		mz += math.sin(a - g + d * 2) * 0.00102
		mz -= math.sin(a * 2 - b * 2 - n) * 0.00106
		mz -= math.sin(a * 2 + n) * 0.00106
		mz -= math.sin(a + b * 2 - d * 2) * 0.00109
		mz -= math.sin(b * 2 - d + n * 2) * 0.0011
		mz += math.sin(d * 4) * 0.00112
		mz -= math.sin(a * 2 - n) * 0.00122
		mz -= math.sin(a * 2 + b * 2 + n) * 0.00122
		mz += math.sin(g + b * 2 - d * 2 + n * 2) * 0.00149
		mz -= math.sin(a * 2 - d * 4) * 0.00157
		mz += math.sin(a + g + b * 2 - d * 2 + n * 2) * 0.00171
		mz -= math.sin(a * 2 + g - d * 2) * 0.00175
		mz -= math.sin(g * 2 - d * 2) * 0.0019
		mz += math.sin(a + e * 16 - w * 18) * 0.00193
		mz += math.sin(a * 2 + b * 2 - d * 2 + n * 2) * 0.00194
		mz += math.sin(g - d * 2 - n) * 0.00201
		mz += math.sin(g + b * 2 - d * 2 + n) * 0.00201
		mz -= math.sin(a + g * 2 - d * 2) * 0.00207
		mz -= math.sin(g * 2) * 0.0021
		mz -= math.sin(d * 2 - n) * 0.00213
		mz -= math.sin(b * 2 + d * 2 + n) * 0.00213
		mz -= math.sin(a * 3 - d * 2) * 0.00215
		mz -= math.sin(a - d * 4) * 0.00247
		mz -= math.sin(a - b * 2 + d * 2) * 0.00253
		mz += ta * 0.00279 * math.sin(b * 2 + n * 2)
		mz -= math.sin(a * 2 + b * 2 + n * 2) * 0.0028
		mz += math.sin(a * 3) * 0.00312
		mz -= math.sin(a + b * 2) * 0.00317
		mz -= math.sin(a + e * 16 - w * 18) * 0.0035
		mz += math.sin(g + b * 2 + n * 2) * 0.0039
		mz += math.sin(g - b * 2 - n * 2) * 0.00413
		mz -= math.sin(n * 2) * 0.0049
		mz -= math.sin(b * 2 + d * 2 + n * 2) * 0.00491
		mz += math.sin(g + d) * 0.00504
		mz += math.sin(a - d) * 0.00516
		mz -= math.sin(g + d * 2) * 0.00621
		mz += math.sin(a - b * 2 - d * 2 - n) * 0.00648
		mz += math.sin(a - d * 2 + n) * 0.00648
		mz += math.sin(a - g - d * 2) * 0.007
		mz += math.sin(a + d * 2) * 0.01122
		mz += math.sin(a - d * 2 - n) * 0.0141
		mz += math.sin(a + b * 2 - d * 2 + n) * 0.0141
		mz += math.sin(a - b * 2) * 0.01424
		mz += math.sin(a - b * 2 - d * 2 - n * 2) * 0.01506
		mz -= math.sin(b * 2 - d * 2) * 0.01567
		mz += math.sin(b * 2 - d * 2 + n * 2) * 0.02077
		mz -= math.sin(a + g) * 0.02527
		mz -= math.sin(a - n) * 0.02952
		mz -= math.sin(a + b * 2 + n) * 0.02952
		mz -= math.sin(d) * 0.03487
		mz += math.sin(a - g) * 0.03684
		mz -= math.sin(d * 2 + n) * 0.03983
		mz += math.sin(b * 2 - d * 2 + n) * 0.03983
		mz += math.sin(a + b * 2 - d * 2 + n * 2) * 0.04037
		mz += math.sin(a * 2) * 0.04221
		mz -= math.sin(g - d * 2) * 0.04273
		mz -= math.sin(a * 2 - d * 2) * 0.05566
		mz -= math.sin(a + g - d * 2) * 0.05697
		mz -= math.sin(a + b * 2 + n * 2) * 0.06846
		mz -= math.sin(a - b * 2 - n) * 0.08724
		mz -= math.sin(a + n) * 0.08724
		mz -= math.sin(b * 2) * 0.11463
		mz -= math.sin(g) * 0.18647
		mz -= math.sin(a - b * 2 - n * 2) * 0.20417
		mz += math.sin(d * 2) * 0.59616
		mz += math.sin(n) * 1.07142
		mz -= math.sin(b * 2 + n) * 1.07447
		mz -= math.sin(a - d * 2) * 1.28658
		mz -= math.sin(b * 2 + n * 2) * 2.4797
		mz += math.sin(a) * 6.32962

		return mx, my, mz
