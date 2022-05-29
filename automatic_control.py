#!/usr/bin/env python3.8

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import copy
import datetime
import glob
import logging
import math
import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.dummy import Process
import os
import threading
import numpy.random as random
import re
import sys
import weakref

# Ours

from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from cv2 import cv2 as cv

sys.path.append('./d-star-lite')
from grid import GridWorld
from utils import stateNameToCoords
from d_star_lite import initDStarLite, moveAndRescan

try:
    sys.path.append('/home/youssef/Apps/CARLA_0.9.12/PythonAPI/carla')
except IndexError:
    pass

import matplotlib.pyplot as plt
from multiprocessing import Value, process
# ----

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
"""
try:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
"""
import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x)
                    for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(
                lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


# Birdeye view variables
birdeye_width = 60
birdeye_height = 60
birdeye_ppm = 3
# -----

# Path Planner variables
grid_width = 60
grid_height = 60
grid_actor_start_pos = [(grid_width / 2) - 1, (grid_height / 2) - 1]
grid_actor_start_str = "x" + \
    str(int(grid_actor_start_pos[0])) + "y" + str(int(grid_actor_start_pos[1]))
grid_viewing_range = (birdeye_width / birdeye_ppm) / 2
# -----


def draw_waypoints(world, waypoint, depth=6):
    if depth < 0:
        return
    for w in waypoint.next(4.0):
        t = w.transform
        begin = t.location + carla.Location(z=0.5)
        angle = math.radians(t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)
        draw_waypoints(world, w, depth - 1)


# Function to get nearest in X and Y


def find_nearest_xy(array, valueX, valueY):
    array = np.asarray(array)
    x = np.abs(array[:, 0] - valueX)
    # print(x)
    # l = np.where(np.logical_and(x >= -0.25, x <= 0.25))
    y = np.abs(array[:, 1] - valueY)
    idx = (x + y).argmin()
    return idx
# -----------


def path_plan(goal_bird,):
    graph = GridWorld(grid_width, grid_height)
    s_start = 'x' + str(int(birdeye_height - 1)) + \
        'y' + str(int(birdeye_width / 2 - 1))
    s_goal = 'x' + str(int(goal_bird[0])) + \
        'y' + str(int(goal_bird[1]))
    graph.setStart(s_start)
    graph.setGoal(s_goal)
    k_m = 0
    s_last = s_start
    queue = []
    s_current = s_start
    s_new = s_current
    graph, queue, k_m = initDStarLite(
        graph, queue, s_start, s_goal, k_m)
    path_plan_points_temp = np.empty(shape=(0, 2))
    while s_new != 'goal':
        current_tup = stateNameToCoords(s_current)
        path_plan_points_temp = np.append(
            path_plan_points_temp, np.array([current_tup]), axis=0)
        s_new, k_m = moveAndRescan(
            graph, queue, s_current, grid_viewing_range, k_m)
        s_current = s_new
    existing_shm = shared_memory.SharedMemory(name='shared_mem')
    shared_path = np.ndarray(
        (320, 2), dtype=np.float64, buffer=existing_shm.buf)
    shared_path[:] = np.inf
    shared_path = np.ndarray(
        np.shape(path_plan_points_temp), dtype=np.float64, buffer=existing_shm.buf)
    shared_path[:] = path_plan_points_temp[:]
    # return path_plan_points


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        if args.agent == "Basic":
            agent = BasicAgent(world.player)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Get actor info
        actor_spawn_transform = world.player.get_transform()
        actor_spawn_loc = actor_spawn_transform.location
        actor_spawn_rotation = actor_spawn_transform.rotation
        # ------

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        agent.set_destination(random.choice(spawn_points).location)
        # ---

        # Birdeye view init
        birdview_producer = BirdViewProducer(
            client,  # carla.Client
            target_size=PixelDimensions(
                width=birdeye_width, height=birdeye_height),
            pixels_per_meter=birdeye_ppm,
            crop_type=BirdViewCropType.FRONT_AREA_ONLY)
        # ----

        all_waypoints = world.map.generate_waypoints(1)
        all_waypoints_xy = np.empty(shape=(0, 2))
        for i, val in enumerate(all_waypoints):
            all_waypoints_xy = np.append(all_waypoints_xy, np.array(
                [[val.transform.location.x, val.transform.location.y]]), axis=0)

        # Thread creation
        proc = None
        alive = False
        path_plan_points = np.empty(shape=(0, 2))
        # --------

        shm = shared_memory.SharedMemory(
            name='shared_mem', create=True, size=5120)
        shared_path = np.ndarray((320, 2), dtype=np.float64, buffer=shm.buf)
        frame_counter = 0

        clock = pygame.time.Clock()

        while True:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

             # Our
            birdview = birdview_producer.produce(agent_vehicle=world.player)
            road_mask = birdview[0]
            vehicles_mask = birdview[3]
            ped_mask = birdview[8]
            cenetrs_mask = birdview[2]

            actor_loc = world.player.get_transform().location

            """
            for i, row in enumerate(road_mask):
                left_flag = False
                right_flag = False
                if np.count_nonzero(row) > 0:
                    if row[0] == 1:
                        left_flag = True
                    if row[len(row) - 1]:
                        right_flag = True
                    if left_flag and right_flag or right_flag:
                        found_black = False
                        row_num = i
                        col_num = np.shape(road_mask)[1] - 1
                        while not found_black:
                            if road_mask[row_num][col_num] == 0:
                                row_num -= 1
                                break
                            if (row_num > np.shape(road_mask)[0]):
                                row_num += 1
                        row_num -= 15
                    elif left_flag:
                        row_num = i + 15
                        col_num = 0
                    else:
                        found_black = False
                        row_num = 0
                        col_num = np.shape(road_mask)[1] - 1
                        while not found_black:
                            if road_mask[row_num][col_num] == 1:
                                break
                            col_num -= 1
                        col_num -= 15
                    break
                """
            """
            origin_bird = np.array(
                [actor_loc.x - ((birdeye_width / 2) / birdeye_ppm), actor_loc.y + ((birdeye_height / 2) / birdeye_ppm)])

            # Puts waypoyoints on the birdeye view and addes thier indecies to an array
            
            waypoints_indcies = np.empty(shape=(0, 2))
            for i in range(0, np.shape(road_mask)[0], 18):
                row = road_mask[i]
                road_seg_width = np.count_nonzero(row)
                nt = 0
                no_of_p = 0
                for j, val in enumerate(row):
                    # print(val)
                    if val:
                        nt += 1
                        if nt == 1:
                            no_of_p = j + road_seg_width
                        no_of_p -= 1
                        if no_of_p >= 0:
                            if nt == 12 or (nt - 12) % 18 == 0:
                                waypoints_indcies = np.append(
                                    waypoints_indcies, np.array([[i, j]]), axis=0)
                        else:
                            break

            road_mask_clone = np.copy(road_mask)
            for i, val in enumerate(waypoints_indcies):
                top_left = (val - 9).astype(int)
                for j in range(top_left[0], top_left[0] + 18):
                    flag = False
                    if j > birdeye_width - 1:
                        break
                    for k in range(top_left[1], top_left[1] + 18):
                        if k > birdeye_width - 1:
                            break
                        if road_mask[j, k]:
                            if vehicles_mask[j][k] or ped_mask[j][k]:
                                rows = np.arange(
                                    top_left[0], top_left[0] + 18)
                                cols = np.arange(
                                    top_left[1], top_left[1] + 18)
                                for l in range(rows):
                                    road_mask_clone[rows[l], cols] = 0
                                flag = True
                                break
                    if flag:
                        break

            false_indcies = np.empty(shape=(0, 2))
            for i in range(int(birdeye_ppm / 2), np.shape(road_mask_clone)[0], int(birdeye_ppm)):
                row = road_mask[i]
                for j in range(int(birdeye_ppm / 2), np.shape(road_mask_clone)[1], int(birdeye_ppm)):
                    if not road_mask_clone[i][j]:
                        false_indcies = np.append(
                            false_indcies, np.array([[i, j]]), axis=0)

            theta = world.imu_sensor.compass lalalalal

            # from birdeye to carla
            translated_waypoints = np.array(
                [(origin_bird[0] + (false_indcies[:, 0] / birdeye_ppm)), (origin_bird[1] - (false_indcies[:, 1] / birdeye_ppm))]).T

            c, s = np.cos(theta), np.sin(theta)
            r = np.array([[c, -s], [s, c]])

            rotated_waypoints = np.add(
                r.dot(
                    np.add(
                        translated_waypoints,
                        np.array(
                            [[-actor_loc.x,
                             -actor_loc.y]]
                        )
                    ).T)
                .T, np.array([[actor_loc.x, actor_loc.y]])
            )

            grid_waypoints = np.around(np.array(
                [rotated_waypoints[:, 0] - grid_origin_carla[0], grid_origin_carla[1] - rotated_waypoints[:, 1]]).T).astype(int)

            for i, val in enumerate(grid_waypoints):
                if(val[0] < grid_width and val[1] < grid_height and val[0] >= 0 and val[1] >= 0):
                    grid.cells[val[0]][val[1]] = -1

                # s_new, k_m = moveAndRescan(
                #    grid, queue, s_current, grid_viewing_range, k_m)
                #s_current = s_new
            """
            center_1d = np.empty(shape=(0, 2))
            for i, row in enumerate(cenetrs_mask):
                for j, val in enumerate(row):
                    if val:
                        center_1d = np.append(
                            center_1d, np.array([[i, j]]), axis=0)

            our_center = np.empty(shape=(0, 2))
            center_1d = center_1d.astype(int)
            center_col = birdeye_width / 2 - 1
            for i in range(birdeye_height - 3 * birdeye_ppm, 0, -birdeye_ppm):
                idx = find_nearest_xy(center_1d, i, center_col)
                center_col = center_1d[idx][1]
                our_center = np.append(
                    our_center, np.array([center_1d[idx]]), axis=0)
                center_1d = np.delete(center_1d, idx, axis=0)

            our_center = our_center.astype(int)
            goal_bird = our_center[np.argmin(our_center[:, 0])]

            waypoints_indcies = np.empty(shape=(0, 2))
            lane_width = math.ceil(3.6 * birdeye_ppm)
            for i in range(0, np.shape(road_mask)[0], lane_width):
                row = road_mask[i]
                road_seg_width = np.count_nonzero(row)
                nt = 0
                no_of_p = 0
                for j, val in enumerate(row):
                    # print(val)
                    if val:
                        nt += 1
                        if nt == 1:
                            no_of_p = j + road_seg_width
                        no_of_p -= 1
                        if no_of_p >= 0:
                            if nt == math.ceil(lane_width / 2) or (nt - math.ceil(lane_width / 2)) % lane_width == 0:
                                waypoints_indcies = np.append(
                                    waypoints_indcies, np.array([[i, j]]), axis=0)
                        else:
                            break

            road_mask_clone = np.copy(road_mask)
            for i, val in enumerate(waypoints_indcies):
                top_left = (val - math.floor(lane_width / 2)).astype(int)
                for j in range(top_left[0], top_left[0] + lane_width):
                    flag = False
                    if j > birdeye_width - 1:
                        break
                    for k in range(top_left[1], top_left[1] + lane_width):
                        if k > birdeye_width - 1:
                            break
                        if road_mask[j, k]:
                            if vehicles_mask[j][k] or ped_mask[j][k]:
                                rows = np.arange(
                                    top_left[0], top_left[0] + lane_width)
                                cols = np.arange(
                                    top_left[1], top_left[1] + lane_width)
                                for l in range(rows):
                                    road_mask_clone[rows[l], cols] = 0
                                flag = True
                                break
                    if flag:
                        break

            false_indcies = np.empty(shape=(0, 2))
            for i in range(int(birdeye_ppm / 2), np.shape(road_mask_clone)[0], int(birdeye_ppm)):
                row = road_mask[i]
                for j in range(int(birdeye_ppm / 2), np.shape(road_mask_clone)[1], int(birdeye_ppm)):
                    if not road_mask_clone[i][j]:
                        false_indcies = np.append(
                            false_indcies, np.array([[i, j]]), axis=0)

            rgb = BirdViewProducer.as_rgb(birdview)

            if not alive:
                #shared_path = Value('path', np.empty(shape=(0, 2)))
                proc = Process(target=path_plan, args=[
                    copy.deepcopy(goal_bird)])
                proc.start()
                alive = True
            if len(path_plan_points) != 0:
                for i, val in enumerate(path_plan_points):
                    if val[0] == np.inf or val[1] == np.inf:
                        break
                    rgb[int(val[0])][int(val[1])] = (255, 0, 0)
            # ----
            # Draw Birdeye window

            rgb[goal_bird[0]][goal_bird[1]] = (0, 255, 0)
            bgr = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
            rgb_canvas = np.zeros(shape=(120, 120, 3), dtype=np.uint8)
            rgb_canvas[np.nonzero(road_mask)] = (255, 255, 255)
            edges = cv.Canny(rgb_canvas, 50, 200)
            cv.imshow("BirdView RGB", bgr)

            # Birdeye window end
            # ------

            if proc is not None and not proc.is_alive():
                print("Balllop")
                alive = False
                path_plan_points = copy.deepcopy(shared_path)

                print(path_plan_points[0])

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            if agent.done():
                if args.loop:
                    agent.set_destination(random.choice(spawn_points).location)
                    world.hud.notification(
                        "The target has been reached, searching for another target", seconds=4.0)
                    print("The target has been reached, searching for another target")
                else:
                    print("The target has been reached, stopping the simulation")
                    break
            key = cv.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
            control = agent.run_step()
            control.manual_gear_shift = False
            world.player.apply_control(control)

    finally:
        shm.close()
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            cv.destroyAllWindows()
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
