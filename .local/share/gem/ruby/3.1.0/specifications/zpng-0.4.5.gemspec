# -*- encoding: utf-8 -*-
# stub: zpng 0.4.5 ruby lib

Gem::Specification.new do |s|
  s.name = "zpng".freeze
  s.version = "0.4.5".freeze

  s.required_rubygems_version = Gem::Requirement.new(">= 0".freeze) if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib".freeze]
  s.authors = ["Andrey \"Zed\" Zaikin".freeze]
  s.date = "2023-02-19"
  s.email = "zed.0xff@gmail.com".freeze
  s.executables = ["zpng".freeze]
  s.extra_rdoc_files = ["LICENSE.txt".freeze, "README.md".freeze, "README.md.tpl".freeze, "TODO".freeze]
  s.files = ["LICENSE.txt".freeze, "README.md".freeze, "README.md.tpl".freeze, "TODO".freeze, "bin/zpng".freeze]
  s.homepage = "http://github.com/zed-0xff/zpng".freeze
  s.licenses = ["MIT".freeze]
  s.rubygems_version = "3.5.9".freeze
  s.summary = "pure ruby PNG file manipulation & validation".freeze

  s.installed_by_version = "3.5.9".freeze if s.respond_to? :installed_by_version

  s.specification_version = 4

  s.add_runtime_dependency(%q<rainbow>.freeze, ["~> 3.1.1".freeze])
  s.add_development_dependency(%q<rspec>.freeze, ["~> 3.11.0".freeze])
  s.add_development_dependency(%q<rspec-its>.freeze, ["~> 1.3.0".freeze])
  s.add_development_dependency(%q<juwelier>.freeze, ["~> 2.4.9".freeze])
end
