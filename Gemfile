source "https://rubygems.org"

gem "jekyll", "~> 4.3.0"
gem "minima", "~> 2.5"

group :jekyll_plugins do
	gem "jekyll-sitemap"
	gem "jekyll-feed"
	gem "jekyll-scholar"
end

install_if -> { RUBY_PLATFORM =~ %r!mingw|mswin|java! } do
	gem "tzinfo", "~> 1.2"
	gem "tzinfo-data"
end

gem "wdm", "~> 0.1.1", :install_if => Gem.win_platform?
